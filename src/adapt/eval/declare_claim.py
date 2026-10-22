"""Declare the configurations of the final claim, before the final run.

Run with `make declare-claim CONFIGS="name=system:adapter_folder ..." SHIPPED=name`. A
configuration is one adapter, or two that must agree, joined with `+`:

    qwen3-4b-sft-r4-0=qwen3-4b-sft-r4-0:adapters/tax-booking/qwen3-4b-r4-0
    agree=a:adapters/tax-booking/a+b:adapters/tax-booking/b

The adapter folders are hashed over `adapter_config.json` and `adapter_model.safetensors`, so the
claim names the exact weights, and a readme or a tokenizer in the folder does not matter. The
shipped configuration is chosen on dev and dev_real, before the final run. See
`adapt.eval.final_lock` for what the lock then allows. `--status` shows the claim on the current
files and which results exist for it.

`make seal FILE=...` records the SHA-256 of a file and the date in the lock, so that it is provable
that the file was fixed before the decisions that follow. The file is hashed and not read, with
Unix line endings, which is how git stores it (`.gitattributes`), so a Windows checkout and a Linux
checkout give the same hash. A sealed file cannot be sealed again with another hash, and `--status`
says whether it still matches.
"""

import argparse
import hashlib
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from adapt.data.rounds import eval_dir, resolve_round
from adapt.eval.final_lock import (
    CLAIM_SPLITS,
    FinalLockError,
    declare,
    read_claims,
    read_sealed,
    seal,
)
from adapt.eval.runner import (
    ADAPTER_FILES,
    RESULTS_ROOT,
    ROOT,
    final_files,
    hash_directory,
    results_file,
)

LOCK = RESULTS_ROOT / "final.lock.json"


def parse_configuration(text: str) -> tuple[str, list[dict[str, str]]]:
    """`name=system:folder` or `name=system:folder+system:folder` as a name and its adapters."""
    name, separator, rest = text.partition("=")
    if not separator or not name or not rest:
        raise ValueError(f"expected name=system:folder, got {text!r}")
    adapters = []
    for part in rest.split("+"):
        system, separator, folder = part.partition(":")
        if not separator or not system or not folder:
            raise ValueError(f"expected system:folder in {text!r}, got {part!r}")
        path = Path(folder)
        if not (path / "adapter_config.json").exists():
            raise ValueError(f"no adapter in {path}")
        adapters.append({"system": system, "adapter_hash": hash_directory(path, ADAPTER_FILES)})
    return name, adapters


def sealed_hash(path: Path) -> str:
    """The SHA-256 of a file with Unix line endings, the way git stores it."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def seal_file(path: Path, lock: Path = LOCK, today: str | None = None) -> str:
    """Seal `path` in the lock under its name without extension. Returns a line to print."""
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    resolved = path.resolve()
    shown = (
        resolved.relative_to(ROOT.resolve()).as_posix()
        if resolved.is_relative_to(ROOT.resolve())
        else path.as_posix()
    )
    sha256 = sealed_hash(path)
    day = today or date.today().isoformat()
    try:
        new = seal(lock, path.stem, shown, sha256, day)
    except FinalLockError as error:
        raise ValueError(str(error)) from error
    entry = read_sealed(lock)[path.stem]
    word = "sealed" if new else "already sealed"
    return f"{word}: {path.stem} {entry['sha256']} on {entry['sealed']}"


def sealed_lines(lock: Path = LOCK) -> list[str]:
    """One line for each sealed file: whether the file on disk still has the sealed hash."""
    lines = []
    for name, entry in read_sealed(lock).items():
        path = ROOT / entry["file"]
        state = (
            "missing"
            if not path.is_file()
            else "matches"
            if sealed_hash(path) == entry["sha256"]
            else "CHANGED"
        )
        lines.append(f"sealed {entry['sealed']}: {name} {entry['sha256'][:12]} ({state})")
    return lines


def status(platform: str, data_round: int, lock: Path = LOCK) -> str:
    sealed = sealed_lines(lock)
    return "\n".join([_claim_status(platform, data_round, lock), *sealed])


def _claim_status(platform: str, data_round: int, lock: Path) -> str:
    files = final_files(platform, data_round=data_round)
    found = next(
        (c for c in read_claims(lock) if {k: c.get(k) for k in files} == files),
        None,
    )
    if found is None:
        return "no claim is declared for the test and gold files of this round"
    out_dir = eval_dir(data_round)
    lines = [
        f"claim of round {found['round']}, started: {found['started']}, shipped: {found['shipped']}"
    ]
    for config in found["configurations"]:
        lines.append(f"  {config['name']}")
        for adapter in config["adapters"]:
            done = [
                split
                for split in CLAIM_SPLITS
                if results_file(out_dir, platform, adapter["system"], split).exists()
            ]
            missing = [split for split in CLAIM_SPLITS if split not in done]
            lines.append(
                f"    {adapter['system']}: results for {', '.join(done) or 'none'}"
                + (f"; missing {', '.join(missing)}" if missing else "")
            )
    overrides = [r["system"] for r in found["runs"] if r["override"]]
    if overrides:
        lines.append(f"  runs outside the list (override): {', '.join(overrides)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Declare the configurations of the final claim.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--round", type=int, help="round of the data (default: the current one)")
    parser.add_argument("--config", nargs="+", help="name=system:folder[+system:folder] ...")
    parser.add_argument("--shipped", help="name of the configuration that is shipped")
    parser.add_argument("--status", action="store_true", help="show the claim and its results")
    parser.add_argument("--seal", type=Path, help="record the SHA-256 and the date of this file")
    args = parser.parse_args(argv)
    try:
        if args.seal:
            print(seal_file(args.seal))
            return 0
        data_round = resolve_round(args.round)
        if args.status:
            print(status(args.platform, data_round))
            return 0
        if not args.config or not args.shipped:
            parser.error("--config and --shipped are needed to declare a claim")
        configurations = dict(parse_configuration(text) for text in args.config)
        if len(configurations) != len(args.config):
            raise ValueError("two configurations have the same name")
        declare(
            LOCK,
            final_files(args.platform, data_round=data_round),
            data_round,
            configurations,
            args.shipped,
        )
    except ValueError as error:
        sys.exit(f"error: {error}")
    print(status(args.platform, data_round))
    return 0


if __name__ == "__main__":
    sys.exit(main())
