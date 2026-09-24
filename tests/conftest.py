import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture(scope="session")
def raw_df():
    from generate_synthetic_data import generate

    return generate(n_rows=2000, seed=42)
