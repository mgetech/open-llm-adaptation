"""Three jobs for a second, independent author, and a comparison of its labels with ours.

`export` writes one folder: new evaluation records to write, the documents of the evaluation
sets to label (without their labels, under new ids in a mixed order), more training documents to
rewrite, a guide and check programs. `compare` reads the labels and lists where they differ from
ours, so a wrong label or a document that can be read two ways can be found before the sets are
used.

This is done before any result on these sets is used for a decision. The answers are never
training data, and a difference is not a reason to change a label unless a person agrees.
"""

import argparse
import json
import random
import shutil
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from adapt.config import PLATFORMS_DIR, load_platform
from adapt.data.gold import gold_path
from adapt.data.rewrite_export import export_extra

SETS = ("gold", "gold_v2", "dev_real")
TEMPLATES = Path(__file__).resolve().parents[3] / "configs" / "audit"
SOURCES = Path(__file__).resolve().parents[1]
TOOLS = (
    SOURCES / "data" / "audit_check.py",
    SOURCES / "data" / "rewrite_check.py",
    SOURCES / "data" / "records_check.py",
    SOURCES / "vat_check.py",
)
KEY_FILE = "key.json"
NEW_RECORDS = {"first_id": 37, "last_id": 96, "review_documents": 12}
QUOTAS = {
    "layouts": {
        "table_invoice": 18,
        "letter_invoice": 9,
        "till_receipt": 15,
        "short_receipt": 6,
        "credit_note": 6,
        "eu_invoice": 6,
    },
    "minimums": {
        "reverse charge": 9,
        "documents with 19% and 7% items": 15,
        "documents with a 0% item": 6,
        "documents with a discount line": 3,
        "purchases on 0410, 0480 or 0650": 5,
        "private purchases (1800)": 3,
        "documents with 4 or more items": 9,
        "documents with one item": 9,
        "different accounts": 18,
    },
}


def read_set(platform: str, name: str) -> list[dict[str, Any]]:
    path = gold_path(platform, PLATFORMS_DIR, name)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_lines(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def export(
    platform: str,
    seed: int,
    guide: Path,
    out: Path,
    rewrite_dir: Path,
    train_size: int,
    data_seed: int,
) -> dict[str, Any]:
    """Write the folder (and a zip) for the second labeller, and the key that maps new ids to ours.

    The folder has three jobs: new evaluation records to write (A), the documents of the evaluation
    sets to label (B), and the weak-category training documents the main rewrite left out (C)."""
    records = [(name, r) for name in SETS for r in read_set(platform, name)]
    random.Random(f"{seed}/audit").shuffle(records)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rules = load_platform(platform).rules
    accounts = [account.number for account in rules.accounts]

    docs, key = [], {}
    for number, (name, record) in enumerate(records, start=1):
        new_id = f"doc-{number:03d}"
        key[new_id] = {"set": name, "id": record["id"]}
        docs.append(
            {
                "id": new_id,
                "layout": record["layout"],
                "chart": record["chart"],
                "document_text": record["document_text"],
            }
        )
    label = out / "B_label"
    label.mkdir()
    _write_lines(label / "docs.jsonl", docs)
    (label / "config.json").write_text(json.dumps({"accounts": accounts}) + "\n", encoding="utf-8")
    (label / "answers.jsonl").write_text("", encoding="utf-8")

    write = out / "A_write"
    write.mkdir()
    names = sorted({r["target"]["supplier"] for _, r in records})
    config = {"accounts": accounts, **NEW_RECORDS, **QUOTAS, "used_names": names}
    (write / "config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    (write / "new_records.jsonl").write_text("", encoding="utf-8")
    shutil.copyfile(TEMPLATES / "WRITE_GUIDE.md", write / "WRITE_GUIDE.md")

    extra = export_extra(platform, data_seed, train_size, rewrite_dir, out / "C_rewrite")

    tools = out / "tools"
    tools.mkdir()
    for source in TOOLS:
        shutil.copyfile(source, tools / source.name)
    shutil.copyfile(guide, out / "GUIDE.md")
    for name in ("README.txt", "PROMPT_A.txt", "PROMPT_B.txt", "PROMPT_C.txt", "finish.py"):
        shutil.copyfile(TEMPLATES / name, out / name)
    key_path = out.with_name(out.name + "-" + KEY_FILE)
    key_path.write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    archive = out.with_name(out.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(f"{out.name}/C_rewrite/out/", "")
        for path in sorted(p for p in out.rglob("*") if p.is_file()):
            handle.write(path, f"{out.name}/{path.relative_to(out).as_posix()}")
    return {
        "documents": len(docs),
        "rewrite_tasks": extra["tasks"],
        "zip": archive,
        "key": key_path,
    }


def _items(target: Mapping[str, Any]) -> list[tuple[float, str, str]]:
    return sorted((round(i["amount"], 2), i["account"], i["vat_rate"]) for i in target["items"])


def differences(ours: Mapping[str, Any], his: Mapping[str, Any]) -> list[str]:
    """What differs between two labels. An empty list means they agree."""
    found = []
    if ours["needs_review"] != his["needs_review"]:
        word = "review" if ours["needs_review"] else "book"
        found.append(f"review: we say {word}, the labeller says the opposite")
        return found
    if ours["needs_review"]:
        return found
    if ours["reverse_charge"] != his["reverse_charge"]:
        found.append(f"reverse_charge: ours {ours['reverse_charge']}, his {his['reverse_charge']}")
    if round(ours["total"], 2) != round(his["total"], 2):
        found.append(f"total: ours {ours['total']}, his {his['total']}")
    if ours["document_type"] != his["document_type"]:
        found.append(f"document_type: ours {ours['document_type']}, his {his['document_type']}")
    if ours["supplier"].lower() != his["supplier"].lower():
        found.append(f"supplier: ours {ours['supplier']!r}, his {his['supplier']!r}")
    if ours["document_date"] != his["document_date"]:
        found.append(f"date: ours {ours['document_date']}, his {his['document_date']}")
    mine, theirs = _items(ours), _items(his)
    if [i[0] for i in mine] != [i[0] for i in theirs]:
        found.append(f"item amounts: ours {[i[0] for i in mine]}, his {[i[0] for i in theirs]}")
    else:
        for (amount, account, rate), (_, his_account, his_rate) in zip(mine, theirs, strict=True):
            if account != his_account:
                found.append(f"account of {amount}: ours {account}, his {his_account}")
            if rate != his_rate:
                found.append(f"VAT rate of {amount}: ours {rate}, his {his_rate}")
    return found


def compare(platform: str, key_path: Path, answers_path: Path, report_path: Path) -> dict[str, Any]:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    answers = {}
    for line in answers_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("{"):
            row = json.loads(line)
            answers[row["id"]] = row
    records = {name: {r["id"]: r for r in read_set(platform, name)} for name in SETS}
    totals: Counter = Counter()
    agreed: Counter = Counter()
    kinds: Counter = Counter()
    lines = []
    for new_id, origin in key.items():
        record = records[origin["set"]][origin["id"]]
        totals[origin["set"]] += 1
        if new_id not in answers:
            kinds["no answer"] += 1
            continue
        found = differences(record["target"], answers[new_id]["target"])
        if not found:
            agreed[origin["set"]] += 1
            continue
        kinds[found[0].split(":")[0].split(" of ")[0]] += 1
        his = answers[new_id]["target"]
        lines += [
            f"### {origin['id']} ({origin['set']}, {record['layout']})",
            "",
            "```",
            record["document_text"],
            "```",
            "",
            *[f"- {d}" for d in found],
            f"- our reason: {record['target']['reason']}",
            f"- his reason: {his['reason']}",
            "",
        ]
    summary = [
        f"{name}: {agreed[name]} of {totals[name]} agree ({agreed[name] / totals[name]:.0%})"
        for name in SETS
        if totals[name]
    ]
    head = ["# Comparison with the independent labels", "", *[f"- {s}" for s in summary], ""]
    head += [f"- first difference: {kind}: {count}" for kind, count in kinds.most_common()]
    report_path.write_text("\n".join(head + [""] + lines) + "\n", encoding="utf-8")
    return {"summary": summary, "kinds": dict(kinds), "report": report_path}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Independent labels for the evaluation sets.")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("export", help="write the folder for the labeller")
    one.add_argument("--platform", default="tax-booking")
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--guide", type=Path, required=True, help="the labelling guide, a text file")
    one.add_argument("--out", type=Path, default=Path("data/audit/combined"))
    one.add_argument(
        "--rewrite-dir", type=Path, default=Path("data/rewrite/round3"), help="the main rewrite"
    )
    one.add_argument("--train-size", type=int, default=6000)
    one.add_argument("--data-seed", type=int, default=2, help="the seed of the round 3 data")
    two = sub.add_parser("compare", help="compare the answers with our labels")
    two.add_argument("--platform", default="tax-booking")
    two.add_argument("--key", type=Path, default=Path("data/audit/combined-key.json"))
    two.add_argument("--answers", type=Path, required=True)
    two.add_argument("--report", type=Path, default=Path("data/audit/report.md"))
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export(
            args.platform,
            args.seed,
            args.guide,
            args.out,
            args.rewrite_dir,
            args.train_size,
            args.data_seed,
        )
        print(f"{result['documents']} documents to label, {result['rewrite_tasks']} to rewrite")
        print(f"zip: {result['zip']}")
        print(f"key (keep it, do not send it): {result['key']}")
        return
    result = compare(args.platform, args.key, args.answers, args.report)
    print("\n".join(result["summary"]))
    print(f"report: {result['report']}")


if __name__ == "__main__":
    main()
