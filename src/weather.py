"""
Weather-Based Intelligence module for AgriSmart AI.
Fetches a 3-day forecast from the Open-Meteo API (no API key required)
and returns tomorrow's precipitation and temperature data.
"""

import requests


# Open-Meteo free forecast endpoint
_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Variables requested from the daily forecast
_DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
]


def get_weather(lat: float, lon: float) -> dict:
    """
    Fetch tomorrow's weather forecast for a given location.

    Uses the Open-Meteo free API (no API key needed).
    Requests a 3-day daily forecast and returns data for day index 1
    (i.e., tomorrow).

    Args:
        lat (float): Latitude of the location (e.g., 23.03 for Ahmedabad).
        lon (float): Longitude of the location (e.g., 72.58 for Ahmedabad).

    Returns:
        dict: A dictionary with the following keys:
            - "rain_tomorrow_mm" (float): Expected precipitation in millimetres.
            - "max_temp" (float): Expected maximum temperature in °C.
            - "min_temp" (float): Expected minimum temperature in °C.

    Raises:
        ValueError: If latitude or longitude are outside valid ranges.
        ConnectionError: If the API request fails due to a network issue.
        RuntimeError: If the API returns an unexpected or malformed response.
    """
    # --- Input validation ---
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Invalid latitude: {lat}. Must be between -90 and 90.")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Invalid longitude: {lon}. Must be between -180 and 180.")

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(_DAILY_VARS),
        "forecast_days": 3,
        "timezone": "auto",
    }

    # --- API request ---
    try:
        response = requests.get(_OPEN_METEO_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Could not reach Open-Meteo API. Check your internet connection.\n{e}"
        ) from e
    except requests.exceptions.Timeout:
        raise ConnectionError("Open-Meteo API request timed out after 10 seconds.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Open-Meteo API returned an error: {response.status_code} {response.reason}\n{e}"
        ) from e

    # --- Parse response ---
    try:
        data = response.json()
        daily = data["daily"]

        # Index 0 = today, index 1 = tomorrow
        rain_tomorrow_mm = daily["precipitation_sum"][1]
        max_temp = daily["temperature_2m_max"][1]
        min_temp = daily["temperature_2m_min"][1]

    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected API response structure from Open-Meteo.\n"
            f"Raw response: {response.text[:300]}\n{e}"
        ) from e

    return {
        "rain_tomorrow_mm": rain_tomorrow_mm,
        "max_temp": max_temp,
        "min_temp": min_temp,
    }


if __name__ == "__main__":
    # Quick demo: Ahmedabad, Gujarat, India
    TEST_LAT = 23.03
    TEST_LON = 72.58

    result = get_weather(TEST_LAT, TEST_LON)

    print(f"Latitude + Longitude")
    print(f"  ({TEST_LAT}, {TEST_LON})")
    print(f"        |")
    print(f"        v")
    print(f"   Open-Meteo API")
    print(f"        |")
    print(f"        v")
    print(f"   3-day forecast")
    print(f"        |")
    print(f"        v")
    print(f"Tomorrow's:")
    print(f"* Rainfall        : {result['rain_tomorrow_mm']} mm")
    print(f"* Max temperature : {result['max_temp']} deg C")
    print(f"* Min temperature : {result['min_temp']} deg C")
