"""
AgriSmart AI — Crop Recommendation Model Training Script
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns the dataset MUST contain.
REQUIRED_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"
REQUIRED_COLUMNS = REQUIRED_FEATURES + [TARGET_COLUMN]

# Output path for the trained model (relative to *this* script's directory).
MODEL_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT_PATH = MODEL_DIR / "crop_rec_model.pkl"


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------

def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and verify that all required columns are present.

    Parameters
    ----------
    csv_path : Path
        Absolute or relative path to the Crop Recommendation CSV.

    Returns
    -------
    pd.DataFrame
        The validated DataFrame.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If one or more required columns are missing from the CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"The CSV is missing required column(s): {', '.join(sorted(missing))}"
        )

    return df


def train_model(df: pd.DataFrame) -> tuple:
    """Train a Random Forest classifier on the crop recommendation data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the required feature and target columns.

    Returns
    -------
    tuple[RandomForestClassifier, float, str]
        (trained_model, test_accuracy, classification_report_text)
    """
    # Separate features (X) and target (y)
    X = df[REQUIRED_FEATURES]
    y = df[TARGET_COLUMN]

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return clf, accuracy, report


def save_model(clf: RandomForestClassifier, output_path: Path) -> None:
    """Persist the trained model to disk using joblib.

    Parameters
    ----------
    clf : RandomForestClassifier
        The trained classifier.
    output_path : Path
        Destination file path (e.g. ``model/crop_rec_model.pkl``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"\nModel saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest crop recommendation model.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the Crop Recommendation CSV file.",
    )
    args = parser.parse_args()

    csv_path = Path(args.data)

    try:
        # Step 1 — Load & validate
        print(f"Loading dataset from: {csv_path}")
        df = load_and_validate(csv_path)
        print(f"Dataset loaded — {len(df)} rows, {len(df.columns)} columns\n")

        # Step 2 — Train & evaluate
        clf, accuracy, report = train_model(df)

        print(f"\n{'=' * 50}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"{'=' * 50}")
        print("\nClassification Report:\n")
        print(report)

        # Step 3 — Save the trained model
        save_model(clf, MODEL_OUTPUT_PATH)

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
