"""Grounded Gemini explanations for AgriSmart AI results.

This module is an explanation layer above the existing AgriSmart modules. It
sends the farmer's question together with the actual available AgriSmart
results to the Google Gemini API. Gemini is not allowed
to diagnose disease, recalculate recommendations, or invent missing data.

Configure the API key with ``GEMINI_API_KEY`` or pass it explicitly to
``get_farmer_response``. The model can be configured with ``GEMINI_MODEL``.
"""

import json
import os
import sys
from typing import Any

from google import genai
from google.genai import types


# Default to the lighter Flash model for free-tier rate-limit headroom.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_QUESTION_LENGTH = 2000
MAX_CONTEXT_LENGTH = 12000
SUPPORTED_LANGUAGES = {"English", "Gujarati"}
DEVELOPMENT_ERROR_DETAILS = "AGRISMART_DEBUG_GEMINI_ERRORS"

SYSTEM_PROMPT = """You are the AgriSmart Farmer Assistant.

Your job is to explain farm results in simple, practical language that any
farmer can understand — whether they use a smartphone or get help reading
the screen.

Use the supplied AgriSmart context as the source of truth.
Treat a section marked AVAILABLE as usable evidence and a section marked
UNAVAILABLE as missing data. Always answer using every AVAILABLE section;
one unavailable section must never prevent an answer from using other
available sections. If all sections are unavailable, tell the farmer to run
the relevant analyses first.
Do not invent disease diagnoses, weather conditions, irrigation decisions,
sustainability scores, crop recommendations, pesticide names, doses,
chemical treatments, or agricultural claims that are not present in the
context. Do not recalculate or override any result. If information
is missing or unavailable, say so clearly. If the question is outside the
available information, say that it is outside the current data.

Wording guidelines:
- Refer to the data as "your farm data" or "your results", not
  "AgriSmart results" or "available AgriSmart results".
- Present scores naturally, e.g. "Your sustainability score is 74 out of
  100 (Green Guardian level)" rather than exposing raw field names.
- When reporting a water or resource score, add brief actionable meaning,
  e.g. "water-use score is 50/100 — reducing unnecessary irrigation can
  help improve this."
- Do not expose internal field names, variable names, or module names
  to the farmer.

Preserve technical disease and crop names where appropriate. Answer in the
requested language, while keeping technical names recognizable. Keep answers
concise, practical, and farmer-friendly. For a question asking what to do today,
summarize all available results and give complete actionable next steps,
preferably as 3-6 concise bullet points. Do not stop after an introduction.
Do not claim to be a substitute for local agricultural expertise.

Gujarati language guidelines:
- Write in natural, conversational Gujarati as spoken in villages.
- Do not mechanically translate English technical terms word-for-word.
  Instead, use the simplest Gujarati explanation, placing the English
  technical term in parentheses only when it aids recognition, e.g.
  "વહેલો સુકારો (Early Blight)".
- Use familiar farming words: ખેતર (field), પાક (crop), પાણી (water),
  રોગ (disease), ભલામણ (recommendation).
- Keep the tone warm, respectful, and practical.
"""



class FarmerAssistantError(RuntimeError):
    """User-safe error raised for assistant configuration or API failures."""


def _safe_error_detail(error: Exception) -> str:
    """Return a redacted SDK error suitable for local development output."""
    detail = str(error).replace("\r", " ").replace("\n", " ").strip()
    for marker in ("AIza", "Bearer", "GEMINI_API_KEY"):
        if marker in detail:
            detail = detail.replace(marker, "[REDACTED]")
    if len(detail) > 500:
        detail = detail[:500] + "..."
    return detail or "No provider details were returned."


def _validate_inputs(question: str, language: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a question for the Farmer Assistant.")
    question = question.strip()
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Question is too long. Please keep it under {MAX_QUESTION_LENGTH} characters."
        )
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}. Choose English or Gujarati."
        )
    return question


def build_grounded_context(agrismart_context: dict[str, Any]) -> str:
    """Build clearly labeled context from the supplied AgriSmart results.

    Values are copied from the module results without calculation or
    enrichment. Missing sections remain explicitly marked unavailable when
    the caller supplies that marker.
    """
    if not isinstance(agrismart_context, dict):
        raise ValueError("AgriSmart context must be a dictionary.")

    section_names = {
        "disease": "DISEASE",
        "weather": "WEATHER",
        "irrigation": "IRRIGATION",
        "crop_recommendation": "CROP RECOMMENDATION",
        "sustainability": "SUSTAINABILITY",
    }
    sections = []
    available_modules = []
    unavailable_modules = []
    unavailable_messages = {
        "disease": "Disease analysis has not been performed yet.",
        "weather": "Weather data is unavailable.",
        "irrigation": "Irrigation advice is unavailable.",
        "crop_recommendation": "Crop recommendation has not been generated yet.",
        "sustainability": "Sustainability score has not been calculated yet.",
    }
    for key, heading in section_names.items():
        if key not in agrismart_context:
            continue
        value = agrismart_context[key]
        if _has_meaningful_value(value):
            available_modules.append(heading)
            sections.append(
                f"{heading}\nSTATUS: AVAILABLE\n{_format_context_value(value)}"
            )
        else:
            unavailable_modules.append(heading)
            sections.append(
                f"{heading}\nSTATUS: UNAVAILABLE\n"
                f"{unavailable_messages[key]}"
            )

    extra_values = {
        key: value
        for key, value in agrismart_context.items()
        if key not in section_names
    }
    if extra_values:
        sections.append(f"OTHER AGRISMART RESULTS\n{_format_context_value(extra_values)}")

    availability = "AVAILABLE MODULES: " + (
        ", ".join(available_modules) if available_modules else "None"
    )
    availability += "\nUNAVAILABLE MODULES: " + (
        ", ".join(unavailable_modules) if unavailable_modules else "None"
    )
    context = availability + "\n\n" + "\n\n".join(sections)
    if len(context) > MAX_CONTEXT_LENGTH:
        raise ValueError("AgriSmart context is too large to send to the assistant.")
    return context


def _has_meaningful_value(value: Any) -> bool:
    """Return whether a context value contains actual module output."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {
            "",
            "unavailable",
            "not available",
            "none",
            "null",
        }
    if isinstance(value, dict):
        return any(_has_meaningful_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_meaningful_value(item) for item in value)
    return True


def _format_context_value(value: Any, indent: int = 0) -> str:
    """Format supplied context values without adding agricultural meaning."""
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            label = str(key).replace("_", " ").title()
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{label}:\n{_format_context_value(item, indent + 2)}")
            else:
                lines.append(f"{prefix}{label}: {item}")
        return "\n".join(lines) if lines else f"{prefix}Unavailable"
    if isinstance(value, list):
        return "\n".join(f"{prefix}- {item}" for item in value) or f"{prefix}Unavailable"
    return f"{prefix}{value}"


def _get_api_key(api_key: str | None) -> str | None:
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()

    try:
        import streamlit as st

        secret_key = st.secrets.get("GEMINI_API_KEY")
        if isinstance(secret_key, str) and secret_key.strip():
            return secret_key.strip()
    except Exception:
        pass

    configured_key = os.getenv("GEMINI_API_KEY")
    return configured_key.strip() if isinstance(configured_key, str) else configured_key


def _clean_response(response_text: Any) -> str:
    if not isinstance(response_text, str) or not response_text.strip():
        raise FarmerAssistantError("The language model returned an empty response.")
    return response_text.strip()


def _extract_response_text(response: Any) -> str:
    """Extract all available Gemini text, including multiple content parts."""
    response_text = getattr(response, "text", None)
    parts_text = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                parts_text.append(text.strip())
    combined_parts = "\n".join(parts_text).strip()
    if combined_parts and (
        not isinstance(response_text, str)
        or len(combined_parts) > len(response_text.strip())
    ):
        return combined_parts
    return response_text if isinstance(response_text, str) else ""


def _ensure_complete_response(response: Any, extracted: str) -> None:
    """Reject provider output explicitly marked as incomplete."""
    for candidate in getattr(response, "candidates", None) or []:
        finish_reason = str(getattr(candidate, "finish_reason", ""))
        if "MAX_TOKENS" in finish_reason.upper():
            raise FarmerAssistantError(
                "Gemini stopped before completing the answer. Please try again."
            )
    if not extracted.strip():
        raise FarmerAssistantError("The language model returned an empty response.")


def _emit_development_diagnostics(api_key: str, model_name: str, context: str, response: Any, extracted: str) -> None:
    """Log safe Gemini diagnostics when explicitly enabled for development."""
    if os.getenv(DEVELOPMENT_ERROR_DETAILS) != "1":
        return
    candidates = getattr(response, "candidates", None) or []
    part_count = sum(
        len(getattr(getattr(candidate, "content", None), "parts", None) or [])
        for candidate in candidates
    )
    usage = getattr(response, "usage_metadata", None)
    finish_reasons = [
        str(getattr(candidate, "finish_reason", "unknown"))
        for candidate in candidates
    ]
    context_preview = context[:120].replace("\n", " ")
    context_tail = context[-120:].replace("\n", " ")
    print(
        "[Farmer Assistant diagnostics] "
        f"api_key_present={bool(api_key)} model={model_name} "
        f"context_length={len(context)} context_start={context_preview!r} "
        f"context_end={context_tail!r} response_type={type(response).__name__} "
        f"response_text_length={len(getattr(response, 'text', '') or '')} "
        f"candidate_count={len(candidates)} part_count={part_count} "
        f"finish_reasons={finish_reasons} "
        f"thoughts_token_count={getattr(usage, 'thoughts_token_count', None)} "
        f"extracted_length={len(extracted)} response_text_empty={not bool(getattr(response, 'text', None))}",
        file=sys.stderr,
    )


def get_farmer_response(
    question: str,
    agrismart_context: dict[str, Any],
    language: str = "English",
    api_key: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Return a grounded farmer-friendly answer from the configured LLM.

    Args:
        question: The farmer's current question.
        agrismart_context: Actual results from AgriSmart modules. Missing
            results should be represented as ``"Unavailable"``.
        language: Response language, currently ``English`` or ``Gujarati``.
        api_key: Optional API key. If omitted, ``AGRISMART_LLM_API_KEY`` is
            read from the environment.
        conversation_history: Optional current-session user/assistant messages
            for answering follow-up questions.

    Raises:
        ValueError: For empty, oversized, unsupported, or malformed inputs.
        FarmerAssistantError: If the API is not configured or unavailable.
    """
    question = _validate_inputs(question, language)
    grounded_context = build_grounded_context(agrismart_context)
    configured_key = _get_api_key(api_key)
    if not configured_key:
        raise FarmerAssistantError(
            "Farmer Assistant is not configured. Add the required API key to enable GenAI assistance."
        )

    history = []
    if conversation_history:
        history = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in conversation_history[-10:]
            if message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        ]

    current_prompt = (
        f"Requested response language: {language}\n\n"
        f"AgriSmart context (source of truth):\n{grounded_context}\n\n"
        f"Farmer question:\n{question}"
    )
    contents = [
        types.Content(
            role=message["role"],
            parts=[types.Part.from_text(text=message["content"])],
        )
        for message in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=current_prompt)]))

    try:
        client = genai.Client(api_key=configured_key)
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=800,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
    except Exception as exc:
        error_name = type(exc).__name__.lower()
        error_detail = str(exc).lower()
        if (
            "auth" in error_name
            or "permission" in error_name
            or "unauth" in error_name
            or "api key" in error_detail
            or "api_key" in error_detail
            or "invalid argument" in error_detail and "key" in error_detail
        ):
            message = "The Farmer Assistant API key was rejected. Check the Gemini API key configuration."
        elif (
            "quota" in error_name
            or "resourceexhausted" in error_name
            or "ratelimit" in error_name
            or "quota" in error_detail
            or "rate limit" in error_detail
        ):
            message = "The Farmer Assistant service quota or rate limit was reached. Please try again later."
        elif (
            "timeout" in error_name
            or "connection" in error_name
            or "network" in error_name
            or "timeout" in error_detail
            or "connection" in error_detail
        ):
            message = "The Farmer Assistant could not reach Gemini. Check your connection and try again."
        else:
            message = "The Farmer Assistant service is temporarily unavailable. Please try again later."
        if os.getenv(DEVELOPMENT_ERROR_DETAILS) == "1":
            message = f"{message}\nException type: {type(exc).__name__}\nGemini error: {_safe_error_detail(exc)}"
        raise FarmerAssistantError(message) from exc

    extracted_response = _extract_response_text(response)
    _emit_development_diagnostics(
        configured_key,
        model_name,
        grounded_context,
        response,
        extracted_response,
    )
    _ensure_complete_response(response, extracted_response)
    return _clean_response(extracted_response)


if __name__ == "__main__":
    print("Farmer Assistant module loaded. Configure GEMINI_API_KEY to make an API request.")
