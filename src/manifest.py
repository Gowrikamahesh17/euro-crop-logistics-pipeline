"""Run manifest: records what produced a given set of artifacts (data hash,
git commit, seed, timestamp, and step-specific info) so results are
reproducible and auditable rather than "trust me, it ran."
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MANIFEST_PATH = PROJECT_ROOT / "reports" / "run_manifest.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _data_hash() -> str:
    raw_path = PROJECT_ROOT / os.getenv("DATA_RAW_PATH", "data/raw/EuroCrop_agricultural_logistics_dataset.csv")
    if not raw_path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(raw_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_manifest(step: str, extra: dict | None = None) -> dict:
    """Merge new info for `step` into reports/run_manifest.json and write it back."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    manifest["git_commit"] = _git_commit()
    manifest["data_sha256_16"] = _data_hash()
    manifest["random_seed"] = int(os.getenv("RANDOM_SEED", 42))
    manifest.setdefault("steps", {})
    manifest["steps"][step] = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest
