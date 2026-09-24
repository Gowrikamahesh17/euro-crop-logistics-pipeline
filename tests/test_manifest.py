import json


def test_write_manifest_creates_and_merges(tmp_path, monkeypatch):
    import manifest

    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "run_manifest.json")

    manifest.write_manifest(step="preprocess", extra={"n_rows": 100})
    with open(tmp_path / "run_manifest.json") as f:
        data = json.load(f)
    assert data["steps"]["preprocess"]["n_rows"] == 100
    assert "git_commit" in data
    assert "random_seed" in data

    manifest.write_manifest(step="train", extra={"foo": "bar"})
    with open(tmp_path / "run_manifest.json") as f:
        data = json.load(f)
    # Both steps' records persist across calls rather than overwriting each other.
    assert "preprocess" in data["steps"]
    assert data["steps"]["train"]["foo"] == "bar"
