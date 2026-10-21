"""The lock on the final claim.

The test_ood and gold files can only back one claim. The first run of a system on them for
a final result writes a claim to `results/final.lock.json`: the hashes of the test_ood file and
of the gold files, and the system and its adapter. Another system on the same files is
refused, because a second look at the same test documents is not a clean claim. The same
system and adapter may go on, for example after a crash in the middle of the run.

New test_ood or gold files have other hashes, so they open a new claim. `override` runs anyway,
and the claim is written with `"override": true`.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LOCK_NAME = "final.lock.json"
# The splits that make a final claim. The few-shot baseline does not: it is the system the
# claim is compared with, and nothing is chosen with it.
CLAIM_SPLITS = ("test_ood", "gold", "gold_v2")


class FinalLockError(ValueError):
    """A second final claim on files that already back one."""


def lock_file(out_dir: Path, results_root: Path) -> Path | None:
    """The lock file for results written to `out_dir`, or None when `out_dir` is not inside the
    results folder of the repository: a scratch or test run makes no claim."""
    if not out_dir.resolve().is_relative_to(results_root.resolve()):
        return None
    return results_root / LOCK_NAME


def read_claims(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["claims"]


def _write(path: Path, claims: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps({"claims": claims}, indent=2, ensure_ascii=False) + "\n")


def claim(
    path: Path,
    files: Mapping[str, Any],
    system: str,
    adapter_hash: str | None,
    data_round: int,
    override: bool = False,
) -> None:
    """Record a final claim on `files` (`test_ood_sha256` and `gold_sha256`), or refuse it.

    Raises FinalLockError when another system already holds a claim on the same files and
    `override` is not set."""
    claims = read_claims(path)
    same = [c for c in claims if {k: c.get(k) for k in files} == dict(files)]
    mine = any(c["system"] == system and c["adapter_hash"] == adapter_hash for c in same)
    if same and not mine and not override:
        first = same[0]
        raise FinalLockError(
            f"{first['system']} already made the final claim on these test_ood and gold files "
            f"(round {first['round']}, see {path.name}). Another run on them is a second look. "
            "Use new test_ood and gold files, or give the override if you accept that the "
            "result is not a clean claim."
        )
    if mine:
        return
    claims.append(
        {
            **files,
            "system": system,
            "adapter_hash": adapter_hash,
            "round": data_round,
            "override": bool(same) and override,
        }
    )
    _write(path, claims)
