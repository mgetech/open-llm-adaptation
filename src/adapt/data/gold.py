"""Check the gold evaluation files.

Run with `make validate-gold`. Gold labels are not computed by code, so these
checks are the only guard on them. There are two files: `gold.jsonl`, and `gold_v2.jsonl` with
the documents written before any round-2 result existed. The second one is checked when it is
there.

Each line of `platforms/<platform>/evals/<name>.jsonl` is one JSON object with:
- `id`: a unique string;
- `platform` and `chart`: the platform name and the chart of accounts of its rules;
- `layout` (optional): one of the layouts in `adapt.config.Layout`;
- `document_text`: the text the model reads;
- `target`: the expected output, in the shape of the output schema.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, get_args

from jsonschema import Draft202012Validator

from adapt.config import PLATFORMS_DIR, Layout, Platform, load_platform
from adapt.schema import build_output_schema

REQUIRED_FIELDS = ("id", "platform", "chart", "document_text", "target")
OPTIONAL_FIELDS = ("layout",)
GOLD_NAMES = ("gold", "gold_v2")


def gold_path(platform: str, root: Path = PLATFORMS_DIR, name: str = "gold") -> Path:
    return root / platform / "evals" / f"{name}.jsonl"


def _cents(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"))


def _schema_problems(validator: Draft202012Validator, target: Any) -> list[str]:
    problems = []
    for error in sorted(validator.iter_errors(target), key=lambda e: list(map(str, e.path))):
        path = list(error.path)
        if error.validator == "enum" and path and path[-1] == "account":
            problems.append(f"item {path[-2] + 1}: account {error.instance} is not in the chart")
        else:
            where = ".".join(str(part) for part in path) or "target"
            problems.append(f"{where}: {error.message}")
    return problems


def _amount_problems(target: dict[str, Any]) -> list[str]:
    problems = []
    items = target["items"]
    credit = target["document_type"] == "credit_note"
    if items:
        total = sum((_cents(item["amount"]) for item in items), Decimal(0))
        if total != _cents(target["total"]):
            problems.append(f"the items add up to {total}, but the total is {target['total']}")
    elif not target["needs_review"]:
        problems.append("a document without items must have needs_review set")
    # An invoice can hold a discount line, but its total is not negative.
    if (target["total"] > 0 and credit) or (target["total"] < 0 and not credit):
        problems.append("the sign of the total does not fit the document type")
    if target["needs_review"] and not target["reason"].strip():
        problems.append("a document that needs review must give a reason")
    return problems


def check_record(record: Any, validator: Draft202012Validator, platform: Platform) -> list[str]:
    """Return what is wrong with one record, or an empty list."""
    if not isinstance(record, dict):
        return ["the record is not a JSON object"]
    allowed = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS)
    problems = [f"missing field {name}" for name in REQUIRED_FIELDS if name not in record]
    problems += [f"unknown field {name}" for name in record if name not in allowed]
    if problems:
        return problems

    if not isinstance(record["id"], str) or not record["id"]:
        problems.append("id must be a non-empty string")
    if record["platform"] != platform.config.name:
        problems.append(f"platform must be {platform.config.name}")
    if record["chart"] != platform.rules.chart:
        problems.append(f"chart must be {platform.rules.chart}")
    if "layout" in record and record["layout"] not in get_args(Layout):
        problems.append(f"unknown layout {record['layout']!r}")
    if not isinstance(record["document_text"], str) or not record["document_text"].strip():
        problems.append("document_text must be a non-empty string")

    schema_problems = _schema_problems(validator, record["target"])
    if schema_problems:
        return problems + schema_problems
    return problems + _amount_problems(record["target"])


def validate_gold(path: Path, platform: Platform) -> list[str]:
    """Check every line of a gold file. Each problem names the line and the record id."""
    if not path.is_file():
        return [f"{path} does not exist"]
    validator = Draft202012Validator(build_output_schema(platform.rules))
    problems = []
    seen: dict[str, int] = {}
    count = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            problems.append(f"line {number}: empty line")
            continue
        count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"line {number}: not valid JSON ({error.msg})")
            continue
        name = record.get("id", "?") if isinstance(record, dict) else "?"
        for problem in check_record(record, validator, platform):
            problems.append(f"line {number} ({name}): {problem}")
        if isinstance(record, dict) and isinstance(record.get("id"), str):
            if record["id"] in seen:
                problems.append(f"line {number} ({name}): id already used on line {seen[name]}")
            seen.setdefault(record["id"], number)
    if count == 0:
        problems.append(f"{path} has no records")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the gold evaluation files.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument(
        "--path",
        type=Path,
        help="check this file only (default: gold.jsonl, and gold_v2.jsonl if it exists)",
    )
    args = parser.parse_args(argv)

    if args.path:
        paths = [args.path]
    else:
        paths = [gold_path(args.platform)]
        if gold_path(args.platform, name="gold_v2").exists():
            paths.append(gold_path(args.platform, name="gold_v2"))
    platform = load_platform(args.platform)
    failed = False
    for path in paths:
        problems = validate_gold(path, platform)
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            print(f"{path}: {len(problems)} problem(s)", file=sys.stderr)
            failed = True
        else:
            count = len(path.read_text(encoding="utf-8").splitlines())
            print(f"{path}: {count} records, all valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
