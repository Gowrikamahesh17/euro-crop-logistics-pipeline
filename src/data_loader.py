"""Data ingestion: resolves the raw dataset path via .env and loads it."""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """Load the raw EuroCrop CSV using the path configured in .env."""
    raw_path = os.getenv("DATA_RAW_PATH")
    if raw_path is None:
        raise RuntimeError("DATA_RAW_PATH is not set in .env")

    full_path = PROJECT_ROOT / raw_path
    if not full_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {full_path}")

    logger.info("Loading raw data from %s", full_path)
    df = pd.read_csv(full_path)
    logger.info("Loaded raw data: %d rows, %d columns", df.shape[0], df.shape[1])
    return df
