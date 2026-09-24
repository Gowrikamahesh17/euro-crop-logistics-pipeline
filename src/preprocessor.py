"""Preprocessing: feature engineering, encoding, scaling, and train/val/test splits."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

REGRESSION_TARGETS = ["Spoilage_Risk", "Efficiency_Ratio", "Quality_Maintenance_Ratio"]
CLASSIFICATION_TARGET = "Vehicle_Type"
TARGETS = REGRESSION_TARGETS + [CLASSIFICATION_TARGET]

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive time-based features from the event timestamp and harvest date."""
    df = df.rename(columns={df.columns[0]: "Event_Timestamp"})
    df["Event_Timestamp"] = pd.to_datetime(df["Event_Timestamp"])
    df["Harvest_Date"] = pd.to_datetime(df["Harvest_Date"])

    df["Days_Since_Harvest"] = (df["Event_Timestamp"] - df["Harvest_Date"]).dt.days
    df["Event_Month"] = df["Event_Timestamp"].dt.month
    df["Event_Hour"] = df["Event_Timestamp"].dt.hour
    df["Event_Dayofweek"] = df["Event_Timestamp"].dt.dayofweek

    return df.drop(columns=["Event_Timestamp", "Harvest_Date"])


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Crop_Type. Vehicle_Type is left as-is (it's a target)."""
    return pd.get_dummies(df, columns=["Crop_Type"], prefix="Crop_Type")


def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate feature columns from the 4 target columns (drops Vehicle_Type from X)."""
    y = df[TARGETS].copy()
    X = df.drop(columns=TARGETS)
    return X, y


def scale_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on train only, apply to all three splits."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_val, X_test, scaler


def preprocess(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """Run the full preprocessing pipeline: engineer -> encode -> split -> scale.

    Returns a dict of X_train/X_val/X_test/y_train/y_val/y_test and the fitted scaler.
    """
    logger.info("Starting preprocessing: %d rows, %d columns", *df.shape)

    df = engineer_features(df)
    df = encode_categoricals(df)
    X, y = split_features_targets(df)

    # 70/15/15 train/val/test split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed
    )

    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test)
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
    }


def save_processed(splits: dict, out_dir: Path | None = None) -> None:
    """Write each split to data/processed/ as CSV."""
    out_dir = out_dir or (PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed/"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        path = out_dir / f"{name}.csv"
        splits[name].to_csv(path, index=False)
        logger.info("Wrote %s (%d rows) to %s", name, len(splits[name]), path)


def main():
    from data_loader import load_raw_data
    from log_utils import setup_logging
    from manifest import write_manifest

    setup_logging("02_preprocessing")
    df = load_raw_data()
    splits = preprocess(df)
    save_processed(splits)
    write_manifest(
        step="preprocess",
        extra={
            "n_rows": len(df),
            "train_rows": len(splits["X_train"]),
            "val_rows": len(splits["X_val"]),
            "test_rows": len(splits["X_test"]),
            "n_features": splits["X_train"].shape[1],
        },
    )


if __name__ == "__main__":
    main()
