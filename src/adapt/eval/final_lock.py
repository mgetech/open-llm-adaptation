"""The lock on the final claim.

The test_ood and gold files can back one claim. Before the final run, the claim is declared in
`results/final.lock.json`: the hashes of the test_ood file and of the gold files, the
configurations that will be evaluated, and the one that is shipped. Both are chosen on dev and
dev_real. A configuration is one adapter, or two adapters that must agree, each given with its
system name and the hash of its folder.

A run on those files is allowed only for an adapter of a declared configuration. Any other
system or adapter is refused, and so is a run when nothing was declared. The first run marks
the claim as started, and from then on the list and the shipped choice cannot change. New
test_ood or gold files have other hashes, so they need their own declaration. `override` runs
an adapter that is not in the list and writes it to the claim with `"override": true`; it never
changes the shipped choice.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LOCK_NAME = "final.lock.json"
# The splits that make a final claim. The few-shot baseline does not: it is the system the
# claim is compared with, and nothing is chosen with it.
CLAIM_SPLITS = ("test_ood", "gold", "gold_v2")
MAX_ADAPTERS = 2  # a configuration is one adapter, or two that must agree


class FinalLockError(ValueError):
    """A run or a declaration that the lock does not allow."""


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


def _find(claims: Sequence[dict[str, Any]], files: Mapping[str, Any]) -> dict[str, Any] | None:
    for item in claims:
        if {key: item.get(key) for key in files} == dict(files):
            return item
    return None


def _check_configurations(
    configurations: Mapping[str, Sequence[Mapping[str, str]]], shipped: str
) -> None:
    if not configurations:
        raise FinalLockError("a claim needs at least one configuration")
    for name, adapters in configurations.items():
        if not 1 <= len(adapters) <= MAX_ADAPTERS:
            raise FinalLockError(f"configuration {name} must have one or two adapters")
        if len({a["system"] for a in adapters}) != len(adapters):
            raise FinalLockError(f"configuration {name} names the same system twice")
        if any(not a["adapter_hash"] for a in adapters):
            raise FinalLockError(f"configuration {name} has an adapter without a hash")
    if shipped not in configurations:
        raise FinalLockError(f"the shipped configuration {shipped} is not in the list")


def declare(
    path: Path,
    files: Mapping[str, Any],
    data_round: int,
    configurations: Mapping[str, Sequence[Mapping[str, str]]],
    shipped: str,
) -> None:
    """Declare the claim on `files`: the configurations to evaluate and the shipped one.

    A claim that was declared but not started can be declared again. One that has started cannot
    be changed."""
    _check_configurations(configurations, shipped)
    claims = read_claims(path)
    existing = _find(claims, files)
    if existing and existing["started"]:
        raise FinalLockError(
            f"the claim on these test_ood and gold files has started (shipped: "
            f"{existing['shipped']}). The list and the shipped choice cannot change. "
            "Use new test_ood and gold files for another claim."
        )
    declared = {
        **files,
        "round": data_round,
        "configurations": [
            {"name": name, "adapters": [dict(a) for a in adapters]}
            for name, adapters in configurations.items()
        ],
        "shipped": shipped,
        "started": False,
        "runs": [],
    }
    _write(path, [c for c in claims if c is not existing] + [declared])


def start_run(
    path: Path,
    files: Mapping[str, Any],
    system: str,
    adapter_hash: str | None,
    override: bool = False,
) -> None:
    """Allow a run of `system` on the claimed splits, or refuse it. The first allowed run marks
    the claim as started."""
    claims = read_claims(path)
    found = _find(claims, files)
    if found is None:
        raise FinalLockError(
            "no claim is declared for these test_ood and gold files. Declare the "
            "configurations and the shipped one first (make declare-claim)."
        )
    declared = {
        (a["system"], a["adapter_hash"])
        for config in found["configurations"]
        for a in config["adapters"]
    }
    allowed = (system, adapter_hash) in declared
    if not allowed and not override:
        names = ", ".join(a for a, _ in sorted(declared))
        raise FinalLockError(
            f"{system} is not in the claim declared for these test_ood and gold files "
            f"(declared: {names}; shipped: {found['shipped']}). Give the override to run it "
            "anyway; the result is then not a clean claim."
        )
    run = {"system": system, "adapter_hash": adapter_hash, "override": not allowed}
    if found["started"] and run in found["runs"]:
        return
    found["started"] = True
    if run not in found["runs"]:
        found["runs"].append(run)
    _write(path, claims)
