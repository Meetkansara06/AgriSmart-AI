"""Predict the best crop given soil and weather conditions."""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "crop_rec_model.pkl"

# Feature order must match the training feature order exactly.
FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# ---------------------------------------------------------------------------
# Prediction function (importable API)
# ---------------------------------------------------------------------------

def predict_crop(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
) -> str:
    """Return the recommended crop for the given soil and weather parameters.

    The function loads ``crop_rec_model.pkl`` (produced by ``train_crop.py``)
    and feeds the seven input features through the Random Forest classifier.

    Parameters
    ----------
    N : float
        Ratio of Nitrogen content in the soil.
    P : float
        Ratio of Phosphorus content in the soil.
    K : float
        Ratio of Potassium content in the soil.
    temperature : float
        Average temperature in °C.
    humidity : float
        Relative humidity in %.
    ph : float
        Soil pH value.
    rainfall : float
        Rainfall in mm.

    Returns
    -------
    str
        Name of the recommended crop (e.g. ``"rice"``).

    Raises
    ------
    FileNotFoundError
        If the trained model file is not found at the expected path.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Please run train_crop.py first to generate the model."
        )

    clf = joblib.load(MODEL_PATH)

    # Preserve the feature names used when the Random Forest was trained.
    features = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=FEATURE_NAMES,
    )
    prediction = clf.predict(features)

    return prediction[0]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the best crop given soil and weather conditions.",
    )
    parser.add_argument("--N", type=float, required=True,
                        help="Nitrogen content in soil")
    parser.add_argument("--P", type=float, required=True,
                        help="Phosphorus content in soil")
    parser.add_argument("--K", type=float, required=True,
                        help="Potassium content in soil")
    parser.add_argument("--temperature", type=float, required=True,
                        help="Average temperature (°C)")
    parser.add_argument("--humidity", type=float, required=True,
                        help="Relative humidity (%)")
    parser.add_argument("--ph", type=float, required=True,
                        help="Soil pH value")
    parser.add_argument("--rainfall", type=float, required=True,
                        help="Rainfall (mm)")

    args = parser.parse_args()

    try:
        crop = predict_crop(
            N=args.N,
            P=args.P,
            K=args.K,
            temperature=args.temperature,
            humidity=args.humidity,
            ph=args.ph,
            rainfall=args.rainfall,
        )
        print(f"Recommended crop: {crop}")

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Prediction failed — {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
