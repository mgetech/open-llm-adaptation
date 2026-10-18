"""The committed data must be what the generator makes from the seed in the manifest.

Earlier rounds are kept in `round<N>/` folders. The generator has changed since then, so they
are checked by their hashes only.
"""

import hashlib
import json
from pathlib import Path

import pytest

from adapt.config import load_platform
from adapt.data.split import GENERATOR_VERSION, build_splits, write_data

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "results" / "data_manifest.json"


def test_committed_data_matches_the_generator(tmp_path):
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if committed.get("generator", 1) != GENERATOR_VERSION:
        pytest.skip("the current files come from an older generator and are rebuilt with it")
    platform = committed["platform"]

    splits = build_splits(load_platform(platform), committed["seed"])
    write_data(splits, platform, committed["seed"], tmp_path / "data", tmp_path / "manifest.json")

    rebuilt = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert rebuilt == {key: value for key, value in committed.items() if key != "rounds"}
    for entry in committed["splits"].values():
        expected = (tmp_path / "data" / entry["file"]).read_bytes()
        assert (ROOT / "data" / entry["file"]).read_bytes() == expected, entry["file"]


def test_the_kept_rounds_match_their_hashes():
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rounds = committed["rounds"]
    assert [entry["round"] for entry in rounds] == list(range(1, len(rounds) + 1))
    for entry in rounds:
        assert set(entry["splits"]) == set(committed["splits"])
        for split, kept in entry["splits"].items():
            data = (ROOT / "data" / kept["file"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == kept["sha256"], kept["file"]
            assert len(data.decode("utf-8").splitlines()) == kept["count"], kept["file"]
            assert f"round{entry['round']}/{split}.jsonl" in kept["file"]
