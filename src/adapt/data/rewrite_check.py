"""Check rewritten documents and hand out the next ones to rewrite.

This file uses only the standard library and runs on Python 3.9 or newer, so the person who
writes the rewrites can run it from the folder of the task without installing anything.

Files in the working folder:
- `config.json`: `batch_size` and `max_attempts`.
- `tasks.jsonl`: every document to rewrite, one JSON object per line.
- `pilot.jsonl`: the documents of the first test (their ids are also in `tasks.jsonl`).
- `out/*.jsonl`: the replies. Every line is `{"id": ..., "document_text": ...}`.

Commands:
- `next [--pilot]`: check all replies, then write the next documents to `work/current.jsonl` and
  say which file in `out/` the reply goes to. A document that was rejected comes back with the
  reasons. When nothing is left it writes `accepted.jsonl` and `report.txt` and prints DONE.
- `check`: check all replies and write `accepted.jsonl` and `report.txt`.
- `pack`: check, then put `out/`, `accepted.jsonl` and `report.txt` into `results.zip`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import zipfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

MIN_LENGTH = 40
MAX_LENGTH = 1200
CLOSE_TO_DRAFT = 0.9
DEFAULT_BATCH_SIZE = 40
DEFAULT_MAX_ATTEMPTS = 3

MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
ALLOWED_NON_ASCII = "€£"
GERMAN_WORDS = (
    "rechnung",
    "gutschrift",
    "summe",
    "gesamt",
    "betrag",
    "mwst",
    "datum",
    "danke",
    "vielen",
    "kunde",
    "lieferung",
    "steuer",
    "netto",
    "brutto",
    "zahlung",
    "bestellung",
    "artikel",
    "menge",
    "preis",
    "straße",
    "strasse",
)
FOREIGN_CURRENCY = re.compile(
    r"[$£¥]|\b(?:USD|GBP|CHF|JPY|CNY|CAD|AUD|SEK|NOK|DKK|PLN|CZK|HUF|RON|BGN|TRY)\b"
)
CURRENCY_SIGNS = {"USD": ("$",), "GBP": ("£",), "CHF": ()}

# An amount with two decimals in either style, with an optional minus. A part of a date such as
# 14.03.2026 does not match.
MONEY = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:[.,]\d{3})+|\d+)[.,]\d{2}(?!\d)(?![.,]\d)")
SHORT_DATE = re.compile(r"(?:0?[1-9]|[12]\d|3[01])[.](?:0?[1-9]|1[0-2])")
PERCENT = re.compile(r"(\d{1,2}(?:[.,]\d)?)\s*%")
RATE_WORDS = re.compile(r"vat|tax|net|mwst", re.IGNORECASE)
ACCOUNT_NUMBERS = frozenset(
    {
        "0410",
        "0480",
        "0650",
        "1800",
        "2100",
        "3300",
        "3400",
        "4210",
        "4240",
        "4250",
        "4360",
        "4530",
        "4610",
        "4650",
        "4654",
        "4660",
        "4670",
        "4805",
        "4910",
        "4920",
        "4930",
        "4940",
        "4945",
        "4950",
        "4964",
        "4970",
        "4980",
    }
)

ACCOUNT_HINT = re.compile(
    r"\b(?:account|acct|a/c|gl|konto)\b\W{0,3}(?:no\.?|number|nr\.?)?\W{0,3}(\d{4})\b",
    re.IGNORECASE,
)
IDENTIFIER_SHAPES = (
    ("an email address", re.compile(r"\S+@\S+")),
    ("a phone number", re.compile(r"\+\d|(?<!\d)0\d{2,4}[ /-]\d{3,}")),
    ("an IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2}")),
    ("a tax number", re.compile(r"\b\d{2}(?: ?\d{3}){3}\b")),
    ("a long number", re.compile(r"\d{10,}")),
    ("a placeholder", re.compile(r"<[A-Z_]+>")),
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def decimal_of(value: object) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"))


def parse_money(token: str) -> Decimal:
    """`1.234,56` and `1,234.56` both give 1234.56."""
    sign = -1 if token.startswith("-") else 1
    token = token.lstrip("-")
    separator = max(token.rfind("."), token.rfind(","))
    digits = re.sub(r"[.,]", "", token[:separator]) + "." + token[separator + 1 :]
    return sign * Decimal(digits)


def money_forms(amount: Decimal) -> list[str]:
    """The ways an amount can be written, without a sign: 1234.56, 1,234.56, 1.234,56, 1234,56."""
    plain = f"{abs(amount):.2f}"
    grouped = f"{abs(amount):,.2f}"
    swapped = grouped.translate(str.maketrans(",.", ".,"))
    return list(dict.fromkeys([plain, plain.replace(".", ","), grouped, swapped]))


def has_amount(text: str, amount: Decimal) -> bool:
    """Whether the amount is in the text, with a minus sign when it is negative."""
    sign = "-" if amount < 0 else ""
    before = r"(?<![\d.,])" if sign else r"(?<![\d.,-])"
    for form in money_forms(amount):
        if re.search(rf"{before}{sign}{re.escape(form)}(?!\d)(?![.,]\d)", text):
            return True
    return False


def date_patterns(iso: str) -> list[str]:
    year, month, day = (int(part) for part in iso.split("-"))
    name = MONTHS[month - 1]
    names = {name, name[:3], "sept" if name == "september" else name[:3]}
    month_word = "(?:" + "|".join(sorted(names)) + r")\.?"
    d = rf"0?{day}" if day < 10 else str(day)
    m = rf"0?{month}" if month < 10 else str(month)
    y2 = f"{year % 100:02d}"
    return [
        rf"{year}-{month:02d}-{day:02d}",
        rf"{year}/{m}/{d}",
        rf"\b{d}\.{m}\.(?:{year}|{y2})\b",
        rf"\b{d}/{m}/(?:{year}|{y2})\b",
        rf"\b{d}-{m}-(?:{year}|{y2})\b",
        rf"\b{d}(?:st|nd|rd|th)?\s+{month_word}\s*,?\s+{year}\b",
        rf"\b{month_word}\s+{d}(?:st|nd|rd|th)?\s*,?\s+{year}\b",
    ]


def has_date(text: str, iso: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in date_patterns(iso))


def allowed_amounts(task: dict) -> set[Decimal]:
    """Every amount the text may contain: the facts, and unit prices that multiply back exactly."""
    allowed: set[Decimal] = set()
    values = [item["amount"] for item in task["items"]] + [task["total"]]
    for row in task.get("vat") or []:
        values += [row["net"], row["vat"]]
    payment = task.get("payment") or {}
    values += [v for v in (payment.get("cash"), payment.get("change")) if v is not None]
    for value in values:
        allowed.add(abs(decimal_of(value)))
    for item in task["items"]:
        amount = abs(decimal_of(item["amount"]))
        for quantity in range(2, 100):
            unit = amount / quantity
            if unit == unit.quantize(Decimal("0.01")):
                allowed.add(unit.quantize(Decimal("0.01")))
    return allowed


def unexpected_amounts(text: str, allowed: set[Decimal]) -> list[str]:
    found = []
    for match in MONEY.finditer(text):
        token = match.group()
        after = text[match.end() : match.end() + 1]
        if after == "." and SHORT_DATE.fullmatch(token.lstrip("-")):
            continue  # a short date such as 04.02.
        if abs(parse_money(token)) not in allowed:
            found.append(token)
    return found


def without_kept(task: dict, text: str) -> str:
    """The text without the strings that must stay. They may hold amounts, rates or a sentence
    that is meant for a program, and the other checks do not apply to them."""
    for kept in task.get("keep_verbatim") or []:
        text = text.replace(kept, " ")
    return text


def check_rewrite(task: dict, text: object) -> list[str]:
    """The reasons to reject a rewrite of `task`. An empty list means the rewrite is accepted."""
    if not isinstance(text, str):
        return ["document_text is not a string"]
    reasons: list[str] = []
    lowered = text.lower()
    rest = without_kept(task, text)
    if not MIN_LENGTH <= len(text) <= MAX_LENGTH:
        reasons.append(f"length {len(text)} is not between {MIN_LENGTH} and {MAX_LENGTH}")
    bad = sorted({c for c in text if (ord(c) > 127 and c not in ALLOWED_NON_ASCII)})
    if bad:
        reasons.append(f"unusual character {bad[0]!r}")
    if any(ord(c) < 32 and c != "\n" for c in text):
        reasons.append("control character")
    if "**" in text or re.search(r"^\s*#|^\s*\|?[-: ]+\|[-|: ]+$", text, re.M):
        reasons.append("markdown")
    if task["supplier"].lower() not in lowered:
        reasons.append(f"supplier name {task['supplier']!r} is missing")
    if not has_date(text, task["date"]):
        reasons.append(f"date {task['date']} is missing")

    for item in task["items"]:
        amount = decimal_of(item["amount"])
        if not has_amount(text, amount):
            reasons.append(f"item amount {amount} is missing")
    total = decimal_of(task["total"])
    if not has_amount(text, total):
        reasons.append(f"total {total} is missing")
    elif not any(
        "total" in line.lower() and has_amount(line[line.lower().index("total") :], total)
        for line in text.splitlines()
    ):
        reasons.append('the word "Total" must be followed by the total on the same line')
    extra = sorted(set(unexpected_amounts(rest, allowed_amounts(task))))
    if extra:
        reasons.append(f"amount that is not allowed: {', '.join(extra[:3])}")

    for needed in task.get("keep_verbatim") or []:
        if needed not in text:
            reasons.append(f"missing text that must stay: {needed[:60]!r}")
    reasons += rate_problems(task, rest)
    reasons += currency_problems(task, rest)

    if not task["reverse_charge"] and "reverse" in rest.lower():
        reasons.append("reverse charge is mentioned on a document that has none")
    found = [w for w in GERMAN_WORDS if re.search(rf"\b{w}\b", rest.lower())]
    if found:
        reasons.append(f"German word: {found[0]}")
    reasons += identifier_problems(task, text)

    draft = task.get("draft") or ""
    if draft:
        if text.strip() == draft.strip():
            reasons.append("the text is the draft")
        elif difflib.SequenceMatcher(None, draft, text, autojunk=False).ratio() > CLOSE_TO_DRAFT:
            reasons.append("the text is too close to the draft, write it in your own words")
    return reasons


def rate_problems(task: dict, text: str) -> list[str]:
    allowed = {str(row["rate"]) for row in task.get("vat") or []}
    problems = []
    for line in text.splitlines():
        if not RATE_WORDS.search(line):
            continue
        for match in PERCENT.finditer(line):
            rate = match.group(1).replace(",", ".")
            rate = rate[:-2] if rate.endswith(".0") else rate
            if rate not in allowed:
                problems.append(f"VAT rate {rate}% is not allowed on this document")
    return sorted(set(problems))


def currency_problems(task: dict, text: str) -> list[str]:
    code = task["currency"]
    if code == "EUR":
        found = FOREIGN_CURRENCY.search(text)
        return [f"currency {found.group()} on a euro document"] if found else []
    problems = []
    signs = CURRENCY_SIGNS.get(code, ())
    if code not in text and not any(sign in text for sign in signs):
        problems.append(f"currency {code} is missing")
    if "EUR" in text or "€" in text:
        problems.append("euro on a document in another currency")
    return problems


def identifier_problems(task: dict, text: str) -> list[str]:
    rest = without_kept(task, text)
    problems = []
    if any(m.group(1) in ACCOUNT_NUMBERS for m in ACCOUNT_HINT.finditer(rest)):
        problems.append("an account number or a booking hint")
    for what, pattern in IDENTIFIER_SHAPES:
        if pattern.search(rest):
            problems.append(f"{what} that was not in the task")
    return problems


class State:
    """What the replies say so far."""

    def __init__(self, tasks: list[dict], replies: list[tuple[str, list[dict]]]):
        self.tasks = {task["id"]: task for task in tasks}
        self.order = [task["id"] for task in tasks]
        self.accepted: dict[str, str] = {}
        self.attempts: Counter = Counter()
        self.reasons: dict[str, list[str]] = {}
        self.files: dict[str, dict] = {}
        self.ignored = 0
        seen_texts: set[str] = set()
        for name, rows in replies:
            self.files[name] = {"lines": len(rows), "accepted": 0, "reasons": []}
            for row in rows:
                task = self.tasks.get(row.get("id"))
                if task is None or row["id"] in self.accepted:
                    self.ignored += 1
                    continue
                self.attempts[row["id"]] += 1
                text = row.get("document_text")
                problems = check_rewrite(task, text)
                if not problems and text in seen_texts:
                    problems = ["the same text as another document"]
                if problems:
                    self.reasons[row["id"]] = problems
                else:
                    self.accepted[row["id"]] = text
                    seen_texts.add(text)
                    self.reasons.pop(row["id"], None)
                    self.files[name]["accepted"] += 1
                self.files[name]["reasons"] += problems

    def pending(self, max_attempts: int, only: set[str] | None = None) -> list[str]:
        ids = [i for i in self.order if i not in self.accepted and self.attempts[i] < max_attempts]
        if only is not None:
            ids = [i for i in ids if i in only]
        fresh = [i for i in ids if self.attempts[i] == 0]
        return fresh + [i for i in ids if self.attempts[i] > 0]

    def gave_up(self, max_attempts: int) -> list[str]:
        return [
            i for i in self.order if i not in self.accepted and self.attempts[i] >= max_attempts
        ]


def load_config(folder: Path) -> dict:
    config = {"batch_size": DEFAULT_BATCH_SIZE, "max_attempts": DEFAULT_MAX_ATTEMPTS}
    path = folder / "config.json"
    if path.exists():
        config.update(json.loads(path.read_text(encoding="utf-8")))
    return config


def load_state(folder: Path) -> State:
    tasks = read_jsonl(folder / "tasks.jsonl")
    files = sorted((folder / "out").glob("*.jsonl")) if (folder / "out").exists() else []
    return State(tasks, [(f.name, read_jsonl(f)) for f in files])


def summary(state: State, config: dict, only: set[str] | None = None) -> str:
    ids = state.order if only is None else [i for i in state.order if i in only]
    done = sum(i in state.accepted for i in ids)
    pending = len(state.pending(config["max_attempts"], only))
    gave_up = len([i for i in state.gave_up(config["max_attempts"]) if i in ids])
    return f"accepted {done} of {len(ids)}, waiting {pending}, given up {gave_up}"


def write_outputs(folder: Path, state: State, config: dict) -> None:
    lines = [
        json.dumps({"id": i, "document_text": state.accepted[i]}, ensure_ascii=False)
        for i in state.order
        if i in state.accepted
    ]
    (folder / "accepted.jsonl").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    reasons = Counter(
        re.sub(r"[-\d.,]+", "#", reason)[:60]
        for i in state.order
        if i not in state.accepted
        for reason in state.reasons.get(i, [])
    )
    report = [summary(state, config), f"replies ignored (unknown or repeated ids): {state.ignored}"]
    if reasons:
        report.append("reasons of the documents that are still not accepted:")
        report += [f"  {count} x {reason}" for reason, count in reasons.most_common(15)]
    for i in state.gave_up(config["max_attempts"]):
        report.append(f"given up: {i}: {'; '.join(state.reasons.get(i, []))[:200]}")
    (folder / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


def next_batch(folder: Path, pilot: bool) -> int:
    config = load_config(folder)
    state = load_state(folder)
    only = {row["id"] for row in read_jsonl(folder / "pilot.jsonl")} if pilot else None
    write_outputs(folder, state, config)
    pending = state.pending(config["max_attempts"], only)
    print(summary(state, config, only))
    if state.files:
        last = list(state.files)[-1]
        info = state.files[last]
        print(f"last reply {last}: {info['lines']} lines, {info['accepted']} accepted")
        for reason, count in Counter(info["reasons"]).most_common(5):
            print(f"  rejected {count} x {reason}")
    if not pending:
        label = "PILOT DONE" if pilot else "DONE"
        print(f"{label}. Files: accepted.jsonl and report.txt in {folder}")
        return 0
    batch = pending[: config["batch_size"]]
    work = folder / "work"
    work.mkdir(exist_ok=True)
    lines = []
    for doc_id in batch:
        task = dict(state.tasks[doc_id])
        if doc_id in state.reasons:
            task["rejected"] = state.reasons[doc_id]
        lines.append(json.dumps(task, ensure_ascii=False))
    (work / "current.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = folder / "out"
    out.mkdir(exist_ok=True)
    number = len(list(out.glob("*.jsonl"))) + 1
    print(f"{len(batch)} documents are in work/current.jsonl")
    print(f"Write your answer, one JSON line for each document, to out/reply-{number:04d}.jsonl")
    return len(batch)


def pack(folder: Path) -> Path:
    """Write `results.zip` with the replies, the accepted rewrites and the report."""
    state = load_state(folder)
    write_outputs(folder, state, load_config(folder))
    target = folder / "results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((folder / "out").glob("*.jsonl")):
            archive.write(path, f"out/{path.name}")
        for name in ("accepted.jsonl", "report.txt"):
            archive.write(folder / name, name)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check rewritten documents, hand out the next ones."
    )
    parser.add_argument("command", choices=["next", "check", "pack"])
    parser.add_argument("--pilot", action="store_true", help="only the documents of the pilot")
    parser.add_argument("--dir", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "pack":
        target = pack(args.dir)
        print(f"{summary(load_state(args.dir), load_config(args.dir))}. Send this file: {target}")
        return 0
    if args.command == "check":
        state = load_state(args.dir)
        config = load_config(args.dir)
        write_outputs(args.dir, state, config)
        print(summary(state, config))
        return 0
    next_batch(args.dir, args.pilot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
