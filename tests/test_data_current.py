"""The committed data must be exactly what the generator makes from the seed in the manifest."""

import json
from pathlib import Path

from adapt.config import load_platform
from adapt.data.split import build_splits, write_data

ROOT = Path(__file__).resolve().parents[1]


def test_committed_data_matches_the_generator(tmp_path):
    manifest_path = ROOT / "results" / "data_manifest.json"
    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    platform = committed["platform"]

    splits = build_splits(load_platform(platform), committed["seed"])
    write_data(splits, platform, committed["seed"], tmp_path / "data", tmp_path / "manifest.json")

    assert (tmp_path / "manifest.json").read_bytes() == manifest_path.read_bytes()
    for entry in committed["splits"].values():
        expected = (tmp_path / "data" / entry["file"]).read_bytes()
        assert (ROOT / "data" / entry["file"]).read_bytes() == expected, entry["file"]
