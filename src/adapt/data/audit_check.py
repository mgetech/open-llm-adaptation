"""Check the answers of an independent labeller before they are sent back.

This file uses only the standard library and runs on Python 3.9 or newer, so the person who
labels the documents can run it from the folder of the task.

Files in the working folder:
- `docs.jsonl`: the documents, one JSON object per line with `id` and `document_text`.
- `config.json`: the accounts that exist.
- `answers.jsonl`: the answers, one JSON object per line: `{"id": ..., "target": {...}}`.

Commands:
- `check`: check `answers.jsonl` and say what is wrong or missing.
- `pack`: check, and when everything is fine put `answers.jsonl` into `audit_results.zip`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

TARGET_KEYS = {
    "supplier",
    "document_date",
    "document_type",
    "currency",
    "total",
    "items",
    "reverse_charge",
    "needs_review",
    "reason",
}
ITEM_KEYS = {"description", "amount", "account", "vat_rate"}
DOCUMENT_TYPES = ("invoice", "receipt", "credit_note")
VAT_RATES = ("19", "7", "0")
MAX_REASON = 120


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            rows.append({"_broken": line[:60]})
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def target_problems(target: object, accounts: set) -> list[str]:
    if not isinstance(target, dict):
        return ["target is not an object"]
    problems = []
    if set(target) != TARGET_KEYS:
        missing = sorted(TARGET_KEYS - set(target))
        extra = sorted(set(target) - TARGET_KEYS)
        return [f"target keys are wrong (missing {missing}, extra {extra})"]
    try:
        date.fromisoformat(str(target["document_date"]))
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(target["document_date"])):
            raise ValueError
    except ValueError:
        problems.append("document_date must be YYYY-MM-DD")
    if target["document_type"] not in DOCUMENT_TYPES:
        problems.append("document_type must be invoice, receipt or credit_note")
    if not (isinstance(target["currency"], str) and re.fullmatch(r"[A-Z]{3}", target["currency"])):
        problems.append("currency must be three capital letters")
    if not isinstance(target["supplier"], str) or not target["supplier"].strip():
        problems.append("supplier must be a text")
    if not is_number(target["total"]):
        problems.append("total must be a number")
    for key in ("reverse_charge", "needs_review"):
        if not isinstance(target[key], bool):
            problems.append(f"{key} must be true or false")
    reason = target["reason"]
    if not isinstance(reason, str) or len(reason) > MAX_REASON:
        problems.append(f"reason must be a text of at most {MAX_REASON} characters")
    elif target["needs_review"] is True and not reason.strip():
        problems.append("a document that needs review needs a reason")
    items = target["items"]
    if not isinstance(items, list):
        return problems + ["items must be a list"]
    if not items and target["needs_review"] is not True:
        problems.append("a document without items must have needs_review true")
    amounts = []
    for number, item in enumerate(items, start=1):
        if not isinstance(item, dict) or set(item) != ITEM_KEYS:
            problems.append(f"item {number} must have exactly {sorted(ITEM_KEYS)}")
            continue
        if not is_number(item["amount"]):
            problems.append(f"item {number}: amount must be a number")
        else:
            amounts.append(item["amount"])
        if item["account"] not in accounts:
            problems.append(f"item {number}: account {item['account']!r} is not in the list")
        if item["vat_rate"] not in VAT_RATES:
            problems.append(f"item {number}: vat_rate must be '19', '7' or '0'")
    complete = amounts and is_number(target["total"]) and len(amounts) == len(items)
    if complete and abs(round(sum(amounts), 2) - round(target["total"], 2)) > 0.005:
        problems.append(f"the items add up to {round(sum(amounts), 2)}, not to the total")
    return problems


def check_folder(folder: Path) -> tuple[list[str], int, int]:
    """The problems found, the number of documents and the number of good answers."""
    docs = [row["id"] for row in read_jsonl(folder / "docs.jsonl")]
    accounts = set(json.loads((folder / "config.json").read_text(encoding="utf-8"))["accounts"])
    problems: list[str] = []
    seen: dict[str, int] = {}
    good = 0
    for row in read_jsonl(folder / "answers.jsonl"):
        if "_broken" in row:
            problems.append(f"a line is not valid JSON: {row['_broken']!r}")
            continue
        doc_id = row.get("id")
        if doc_id not in docs:
            problems.append(f"unknown id {doc_id!r}")
            continue
        seen[doc_id] = seen.get(doc_id, 0) + 1
        found = target_problems(row.get("target"), accounts)
        if seen[doc_id] > 1:
            problems.append(f"{doc_id}: answered twice")
        elif found:
            problems += [f"{doc_id}: {p}" for p in found]
        else:
            good += 1
    problems += [f"{doc_id}: no answer yet" for doc_id in docs if doc_id not in seen]
    return problems, len(docs), good


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the answers of the labeller.")
    parser.add_argument("command", choices=["check", "pack"])
    parser.add_argument("--dir", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    problems, total, good = check_folder(args.dir)
    for problem in problems[:60]:
        print(problem)
    if len(problems) > 60:
        print(f"... and {len(problems) - 60} more")
    print(f"{good} of {total} documents have a good answer, {len(problems)} problems")
    if args.command == "check":
        return 0 if not problems else 1
    if problems:
        print("Fix the problems first, then run pack again.")
        return 1
    target = args.dir / "audit_results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.dir / "answers.jsonl", "answers.jsonl")
    print(f"Everything is fine. Send this file: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
