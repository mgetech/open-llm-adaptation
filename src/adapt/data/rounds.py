"""Which round of data a run uses, and what a round adds to the names of its output.

The current round is the one in `data/<platform>/`. Earlier rounds are kept in `round<N>/` folders
and listed in `results/data_manifest.json`. Round 1 keeps the plain names (`qwen3-4b-0`). Any
later round adds `-r<N>` after the model (`qwen3-4b-r2-0`), so a run on newer data never replaces
one on older data.

A run that does not name a round uses the current one.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
MANIFEST_NAME = "data_manifest.json"


def manifest_path(data_dir: Path = DATA_DIR) -> Path:
    return data_dir.parent / "results" / MANIFEST_NAME


def _read_manifest(data_dir: Path) -> dict:
    path = manifest_path(data_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def current_round(data_dir: Path = DATA_DIR) -> int:
    """The round of the files in `data/<platform>/`: one more than the rounds that were kept."""
    return len(_read_manifest(data_dir).get("rounds", [])) + 1


def resolve_round(data_round: int | None, data_dir: Path = DATA_DIR) -> int:
    """The round to use: the one asked for, or the current one when none is asked for."""
    current = current_round(data_dir)
    if data_round is None:
        return current
    if not 1 <= data_round <= current:
        raise ValueError(f"round {data_round} does not exist, the rounds are 1 to {current}")
    return data_round


def round_folder(platform: str, data_round: int | None, data_dir: Path = DATA_DIR) -> Path:
    """The folder with the split files of a round."""
    resolved = resolve_round(data_round, data_dir)
    base = data_dir / platform
    return base if resolved == current_round(data_dir) else base / f"round{resolved}"


def round_tag(data_round: int) -> str:
    """What a round adds to a name: nothing for round 1, `-r2` for round 2."""
    return "" if data_round == 1 else f"-r{data_round}"


def generator_version(data_round: int | None, data_dir: Path = DATA_DIR) -> int:
    """The generator version that made a round. The manifests of round 1 do not record it."""
    manifest = _read_manifest(data_dir)
    resolved = resolve_round(data_round, data_dir)
    if resolved == current_round(data_dir):
        return manifest.get("generator", 1)
    entry = next(e for e in manifest["rounds"] if e["round"] == resolved)
    return entry.get("generator", 1)
