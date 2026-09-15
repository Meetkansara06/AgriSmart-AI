"""
Smart Irrigation module for AgriSmart AI.
Provides a simple, rule-based irrigation decision for farmers
based on tomorrow's expected rainfall, crop growth stage, and soil type.

The rainfall value is intended to come from src/weather.py:
    weather = get_weather(lat, lon)
    advice  = irrigation_advice(weather["rain_tomorrow_mm"], crop_stage, soil_type)
"""

# Valid input values
_VALID_CROP_STAGES = {"seedling", "growing", "mature"}
_VALID_SOIL_TYPES  = {"sandy", "loamy", "clay"}


def irrigation_advice(rain_mm: float, crop_stage: str, soil_type: str) -> str:
    """
    Return a farmer-friendly irrigation recommendation based on simple rules.

    The decision is entirely rule-based — no ML model is used.
    Rules are applied in priority order (highest rainfall threshold first).

    Args:
        rain_mm (float):
            Expected rainfall tomorrow in millimetres.
            Obtain this from src/weather.py -> get_weather()["rain_tomorrow_mm"].
            Must be a non-negative number.

        crop_stage (str):
            Current growth stage of the crop.
            Allowed values: "seedling", "growing", "mature"

        soil_type (str):
            Type of soil in the field.
            Allowed values: "sandy", "loamy", "clay"

    Returns:
        str: A short, plain-English irrigation decision for the farmer.

    Raises:
        ValueError: If rain_mm is negative or not numeric, or if crop_stage
                    or soil_type are not one of the accepted values.

    Rules (applied in order):
        1. rain_mm > 10              -> Delay irrigation (heavy rain coming)
        2. rain_mm > 4, sandy soil   -> Light irrigation (sandy soil drains fast)
        3. rain_mm > 4, other soil   -> Skip irrigation (moderate rain is enough)
        4. rain_mm <= 4, seedling    -> Irrigate gently (seedlings need steady moisture)
        5. rain_mm <= 4, other stage -> Irrigate today (crop needs water)
    """
    # --- Input validation ---
    if not isinstance(rain_mm, (int, float)):
        raise ValueError(
            f"rain_mm must be a number, got {type(rain_mm).__name__!r}."
        )
    if rain_mm < 0:
        raise ValueError(
            f"rain_mm cannot be negative, got {rain_mm}."
        )
    if crop_stage not in _VALID_CROP_STAGES:
        raise ValueError(
            f"Invalid crop_stage: {crop_stage!r}. "
            f"Allowed values: {sorted(_VALID_CROP_STAGES)}."
        )
    if soil_type not in _VALID_SOIL_TYPES:
        raise ValueError(
            f"Invalid soil_type: {soil_type!r}. "
            f"Allowed values: {sorted(_VALID_SOIL_TYPES)}."
        )

    # Format rainfall for display — drop unnecessary decimals
    rain_display = int(rain_mm) if rain_mm == int(rain_mm) else round(rain_mm, 1)

    # --- Rule 1: Heavy rain (> 10 mm) ---
    if rain_mm > 10:
        return (
            f"Delay irrigation -- heavy rain expected tomorrow ({rain_display}mm). "
            f"No irrigation needed."
        )

    # --- Rule 2: Moderate rain (> 4 mm) + sandy soil ---
    if rain_mm > 4 and soil_type == "sandy":
        return (
            "Light irrigation recommended -- partial rain expected "
            "but sandy soil drains fast."
        )

    # --- Rule 3: Moderate rain (> 4 mm) + non-sandy soil ---
    if rain_mm > 4:
        return (
            f"Skip irrigation -- moderate rain ({rain_display}mm) expected, "
            f"sufficient for {crop_stage} stage."
        )

    # --- Rule 4: Low rain (<= 4 mm) + seedling ---
    if crop_stage == "seedling":
        return (
            "Irrigate gently -- seedlings need consistent moisture "
            "and no rain is expected."
        )

    # --- Rule 5: Low rain (<= 4 mm) + growing or mature ---
    return (
        f"Irrigate today -- no significant rain expected ({rain_display}mm). "
        f"Crop needs water."
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # Demo: covers all 5 rules + boundary values                         #
    # ------------------------------------------------------------------ #
    test_cases = [
        # (rain_mm, crop_stage, soil_type, label)
        # --- Main rule cases ---
        (20,   "growing",  "loamy", "Heavy rain"),
        (6,    "growing",  "sandy", "Moderate rain + sandy soil"),
        (6,    "growing",  "loamy", "Moderate rain + loamy soil"),
        (2,    "seedling", "loamy", "Low rain + seedling"),
        (2,    "mature",   "loamy", "Low rain + mature"),
        (0,    "growing",  "clay",  "Zero rain"),
        # --- Boundary values ---
        (4,    "growing",  "loamy", "Exactly 4 mm (Rule 4/5 boundary)"),
        (4.1,  "growing",  "loamy", "Just above 4 mm (Rule 3)"),
        (10,   "growing",  "loamy", "Exactly 10 mm (Rule 3)"),
        (10.1, "growing",  "loamy", "Just above 10 mm (Rule 1)"),
    ]

    print("AgriSmart AI -- Smart Irrigation Advisor")
    print("=" * 55)
    for rain, stage, soil, label in test_cases:
        advice = irrigation_advice(rain, stage, soil)
        print(f"\n[{label}]")
        print(f"  Input : rain={rain}mm | stage={stage} | soil={soil}")
        print(f"  Advice: {advice}")
    print("\n" + "=" * 55)
