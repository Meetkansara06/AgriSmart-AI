import json
import os
import unittest
from unittest.mock import Mock, patch

from src.farmer_assistant import (
    DEFAULT_MODEL,
    FarmerAssistantError,
    build_grounded_context,
    get_farmer_response,
)


CONTEXT = {
    "disease": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "model_label": "Tomato___Early_blight",
        "confidence": 91.4,
        "precautions": ["Remove affected leaves."],
    },
    "weather": {
        "rain_tomorrow_mm": 12,
        "max_temperature_c": 30,
        "min_temperature_c": 22,
    },
    "irrigation": "Delay irrigation today.",
    "crop_recommendation": {
        "recommended_crop": "rice",
        "inputs": {"N": 90, "P": 42, "K": 43},
    },
    "sustainability": {
        "score": 72,
        "badge": "Green Guardian",
        "suggestion": "Reduce unnecessary water use.",
    },
}


class MockResponse:
    def __init__(self, text):
        self.text = text


class MockPart:
    def __init__(self, text):
        self.text = text


class MockContent:
    def __init__(self, parts):
        self.parts = parts


class MockCandidate:
    def __init__(self, parts, finish_reason=None):
        self.content = MockContent(parts)
        self.finish_reason = finish_reason


class AuthenticationError(Exception):
    pass


class ResourceExhaustedError(Exception):
    pass


class FarmerAssistantTests(unittest.TestCase):
    def test_valid_context_construction(self):
        context = build_grounded_context(CONTEXT)
        self.assertIn("DISEASE", context)
        self.assertIn("WEATHER", context)
        self.assertIn("Tomato___Early_blight", context)
        self.assertIn("Delay irrigation today.", context)

    def test_missing_context_is_explicit(self):
        context = build_grounded_context({"weather": "Unavailable"})
        self.assertIn("WEATHER", context)
        self.assertIn("STATUS: UNAVAILABLE", context)
        self.assertIn("Weather data is unavailable.", context)
        self.assertNotIn("Tomato", context)

    def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(FarmerAssistantError, "not configured"):
                get_farmer_response("What should I do?", CONTEXT)

    def test_invalid_language(self):
        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            get_farmer_response("What should I do?", CONTEXT, language="Hindi")

    def test_empty_question(self):
        with self.assertRaisesRegex(ValueError, "enter a question"):
            get_farmer_response("   ", CONTEXT)

    def test_overly_long_question(self):
        with self.assertRaisesRegex(ValueError, "too long"):
            get_farmer_response("x" * 2001, CONTEXT)

    def test_malformed_api_response(self):
        response = MockResponse("")
        client = Mock()
        client.models.generate_content.return_value = response
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            with self.assertRaisesRegex(FarmerAssistantError, "empty response"):
                get_farmer_response("Why?", CONTEXT, api_key="test-key")

    def test_http_error(self):
        client = Mock()
        client.models.generate_content.side_effect = AuthenticationError()
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            with self.assertRaisesRegex(FarmerAssistantError, "API key was rejected"):
                get_farmer_response("Why?", CONTEXT, api_key="test-key")

    def test_gemini_quota_error(self):
        client = Mock()
        client.models.generate_content.side_effect = ResourceExhaustedError()
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            with self.assertRaisesRegex(FarmerAssistantError, "quota or rate limit"):
                get_farmer_response("Why?", CONTEXT, api_key="test-key")

    def test_gemini_network_error(self):
        client = Mock()
        client.models.generate_content.side_effect = ConnectionError()
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            with self.assertRaisesRegex(FarmerAssistantError, "could not reach Gemini"):
                get_farmer_response("Why?", CONTEXT, api_key="test-key")

    def test_valid_mocked_api_response(self):
        response = MockResponse("Delay watering today because rain is expected.")
        client = Mock()
        client.models.generate_content.return_value = response
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            answer = get_farmer_response("Should I water today?", CONTEXT, api_key="test-key")
        self.assertEqual(answer, "Delay watering today because rain is expected.")

    def test_normal_response_text_is_extracted(self):
        response = MockResponse("Complete recommendation from response text.")
        client = Mock()
        client.models.generate_content.return_value = response
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            answer = get_farmer_response("What should I do?", CONTEXT, api_key="test-key")
        self.assertEqual(answer, "Complete recommendation from response text.")

    def test_multiple_gemini_text_parts_are_combined(self):
        response = MockResponse("Here is your daily update:")
        response.candidates = [
            MockCandidate([
                MockPart("Here is your daily update:"),
                MockPart("Delay irrigation today because rain is expected."),
                MockPart("Monitor the sustainability recommendation."),
            ])
        ]
        client = Mock()
        client.models.generate_content.return_value = response
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            answer = get_farmer_response("What should I do today?", CONTEXT, api_key="test-key")
        self.assertIn("Delay irrigation today", answer)
        self.assertIn("Monitor the sustainability recommendation", answer)

    def test_empty_gemini_response_is_rejected(self):
        response = MockResponse("")
        response.candidates = []
        client = Mock()
        client.models.generate_content.return_value = response
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            with self.assertRaisesRegex(FarmerAssistantError, "empty response"):
                get_farmer_response("What should I do?", CONTEXT, api_key="test-key")

    def test_max_token_response_is_not_returned_as_truncated_text(self):
        response = MockResponse("Here is your daily update:")
        response.candidates = [MockCandidate([], "MAX_TOKENS")]
        client = Mock()
        client.models.generate_content.return_value = response
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            with self.assertRaisesRegex(FarmerAssistantError, "stopped before completing"):
                get_farmer_response("What should I do?", CONTEXT, api_key="test-key")

    def test_daily_guidance_prompt_requests_complete_action(self):
        response = MockResponse("Use the current results.")
        client = Mock()
        client.models.generate_content.return_value = response
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            get_farmer_response("What should I do today based on my farm results?", CONTEXT, api_key="test-key")
        system_instruction = client.models.generate_content.call_args.kwargs["config"].system_instruction
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertIn("3-6 concise bullet points", system_instruction)
        self.assertIn("Do not stop after an introduction", system_instruction)
        self.assertEqual(config.max_output_tokens, 800)
        self.assertEqual(config.thinking_config.thinking_level.value, "MINIMAL")

    def test_conversation_history_handling(self):
        response = MockResponse("The recommendation is unchanged.")
        client = Mock()
        client.models.generate_content.return_value = response
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            get_farmer_response(
                "Why?",
                CONTEXT,
                api_key="test-key",
                conversation_history=[
                    {"role": "user", "content": "Should I water today?"},
                    {"role": "assistant", "content": "The recommendation says delay."},
                ],
            )
        contents = client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(contents[0].parts[0].text, "Should I water today?")
        self.assertEqual(contents[1].parts[0].text, "The recommendation says delay.")

    def test_gujarati_language_request(self):
        response = MockResponse("આજે પાણી આપવાનું ટાળો.")
        client = unittest.mock.Mock()
        client.models.generate_content.return_value = response
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            answer = get_farmer_response(
                "આજે પાણી આપવું જોઈએ?",
                CONTEXT,
                language="Gujarati",
                api_key="test-key",
            )
        self.assertEqual(answer, "આજે પાણી આપવાનું ટાળો.")
        content = client.models.generate_content.call_args.kwargs["contents"][-1]
        self.assertIn("Gujarati", content.parts[0].text)

    def test_actual_values_are_in_grounded_request(self):
        response = MockResponse("Use the current recommendation.")
        client = unittest.mock.Mock()
        client.models.generate_content.return_value = response
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            get_farmer_response("What should I do today?", CONTEXT, api_key="test-key")
        user_prompt = client.models.generate_content.call_args.kwargs["contents"][-1].parts[0].text
        self.assertIn("Rain Tomorrow Mm: 12", user_prompt)
        self.assertIn("Delay irrigation today.", user_prompt)
        self.assertIn("Green Guardian", user_prompt)
        self.assertIn("rice", user_prompt)

    def test_disease_unavailable_other_results_remain_available(self):
        context = {
            "disease": "Unavailable",
            "weather": {"rain_tomorrow_mm": 12},
            "irrigation": "Delay irrigation today.",
            "sustainability": {"score": 72, "badge": "Green Guardian"},
            "crop_recommendation": "Unavailable",
        }
        client = Mock()
        client.models.generate_content.return_value = MockResponse(
            "Delay watering today because rain is expected."
        )
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            answer = get_farmer_response("What should I do today?", context, api_key="test-key")
        prompt = client.models.generate_content.call_args.kwargs["contents"][-1].parts[0].text
        self.assertIn("Rain Tomorrow Mm: 12", prompt)
        self.assertIn("Delay irrigation today.", prompt)
        self.assertIn("Score: 72", prompt)
        self.assertIn("Disease analysis has not been performed yet.", prompt)
        self.assertEqual(answer, "Delay watering today because rain is expected.")

    def test_crop_unavailable_other_results_remain_available(self):
        context = {
            "disease": {"crop": "Tomato", "disease": "Healthy", "confidence": 98.0},
            "weather": {"rain_tomorrow_mm": 0},
            "irrigation": "Irrigate today.",
            "sustainability": {"score": 88, "badge": "Platinum"},
            "crop_recommendation": "Unavailable",
        }
        client = Mock()
        client.models.generate_content.return_value = MockResponse("Use the current available results.")
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            get_farmer_response("What should I do?", context, api_key="test-key")
        prompt = client.models.generate_content.call_args.kwargs["contents"][-1].parts[0].text
        self.assertIn("Healthy", prompt)
        self.assertIn("Irrigate today.", prompt)
        self.assertIn("Crop recommendation has not been generated yet.", prompt)

    def test_all_modules_unavailable(self):
        context = {
            "disease": "Unavailable",
            "weather": "Unavailable",
            "irrigation": "Unavailable",
            "crop_recommendation": "Unavailable",
            "sustainability": "Unavailable",
        }
        client = Mock()
        client.models.generate_content.return_value = MockResponse(
            "Please run the relevant AgriSmart modules first."
        )
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            get_farmer_response("What should I do?", context, api_key="test-key")
        prompt = client.models.generate_content.call_args.kwargs["contents"][-1].parts[0].text
        self.assertIn("AVAILABLE MODULES: None", prompt)
        self.assertIn("UNAVAILABLE MODULES:", prompt)
        self.assertIn("Disease analysis has not been performed yet.", prompt)

    def test_all_modules_available(self):
        client = Mock()
        client.models.generate_content.return_value = MockResponse("Use all current results.")
        with patch("src.farmer_assistant.genai.Client", return_value=client):
            get_farmer_response("What should I do today?", CONTEXT, api_key="test-key")
        prompt = client.models.generate_content.call_args.kwargs["contents"][-1].parts[0].text
        self.assertIn("AVAILABLE MODULES: DISEASE, WEATHER, IRRIGATION, CROP RECOMMENDATION, SUSTAINABILITY", prompt)
        self.assertIn("UNAVAILABLE MODULES: None", prompt)

    def test_unavailable_data_is_not_fabricated(self):
        context = {"disease": "Unavailable", "weather": {"rain_tomorrow_mm": 12}}
        rendered = build_grounded_context(context)
        self.assertIn("STATUS: UNAVAILABLE", rendered)
        self.assertIn("Disease analysis has not been performed yet.", rendered)
        self.assertNotIn("Early Blight", rendered)
        self.assertIn("Rain Tomorrow Mm: 12", rendered)

    def test_default_model_is_flash_lite(self):
        self.assertEqual(DEFAULT_MODEL, "gemini-3.5-flash-lite")

    def test_api_call_uses_default_model(self):
        client = Mock()
        client.models.generate_content.return_value = MockResponse("OK")
        with patch.dict("os.environ", {}, clear=False):
            env = dict(os.environ)
            env.pop("GEMINI_MODEL", None)
            with patch.dict("os.environ", env, clear=True):
                with patch("src.farmer_assistant.genai.Client", return_value=client):
                    get_farmer_response("What should I do?", CONTEXT, api_key="test-key")
        call_kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-3.5-flash-lite")

    def test_gemini_model_env_var_overrides_default(self):
        client = Mock()
        client.models.generate_content.return_value = MockResponse("OK")
        with patch.dict("os.environ", {"GEMINI_MODEL": "gemini-custom-model"}):
            with patch("src.farmer_assistant.genai.Client", return_value=client):
                get_farmer_response("What should I do?", CONTEXT, api_key="test-key")
        call_kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-custom-model")


if __name__ == "__main__":
    unittest.main()
