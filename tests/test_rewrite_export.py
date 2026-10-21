import json
import subprocess
import sys
import zipfile
from collections import Counter
from decimal import Decimal

import pytest

from adapt.config import load_platform
from adapt.data import rewrite_export
from adapt.data.rewrite_check import check_rewrite
from adapt.data.rewrite_export import choose_pilot, choose_tasks, load_config, make_task
from adapt.data.split import SPLIT_SIZES, build_splits
from fake_rewriter import rewrite

SEED = 3
LABEL_KEYS = {"target", "account", "accounts", "needs_review", "reason", "expected_review"}


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def splits(platform):
    return build_splits(platform, SEED, SPLIT_SIZES, keep_clean=True)


@pytest.fixture(scope="module")
def tasks(splits, config):
    return choose_tasks(splits, 1.0, SEED, config)


@pytest.fixture(scope="module")
def records(splits):
    return {r["id"]: r for rows in splits.values() for r in rows}


def test_the_config_loads_and_has_styles_for_every_kind(config):
    kinds = {kind for style in config["styles"] for kind in style["kinds"]}
    assert kinds == {"receipt", "invoice", "credit_note"}
    assert config["batch_size"] == 40 and config["max_attempts"] == 3
    assert {s["foreign"] for s in config["styles"]} == {"only", "never", "any"}


def test_a_task_has_the_facts_and_no_label(tasks):
    assert tasks
    for task in tasks:
        assert LABEL_KEYS.isdisjoint(task)
        assert set(task) == {
            "id", "style", "kind", "supplier", "date", "currency", "reverse_charge", "items",
            "total", "vat", "payment", "keep_verbatim", "instructions", "draft", "draft_sha256",
        }  # fmt: skip
        assert all(set(item) == {"description", "amount"} for item in task["items"])


def test_the_share_is_the_same_in_every_split(splits, config):
    for share in (0.3, 0.6):
        chosen = choose_tasks(splits, share, SEED, config)
        for name, rows in splits.items():
            eligible = [r for r in rows if not r["canary"] and r["category"] not in
                        set(config["never_rewritten"])]  # fmt: skip
            taken = sum(t["id"].startswith(f"{name}-") for t in chosen)
            assert abs(taken / len(eligible) - share) < 0.1, (name, share)


def test_canaries_and_damaged_documents_are_never_sent(tasks, records, config):
    skipped = set(config["never_rewritten"])
    for task in tasks:
        record = records[task["id"]]
        assert not record["canary"]
        assert record["category"] not in skipped


def test_the_choice_is_fixed_by_the_seed(splits, config):
    first = choose_tasks(splits, 0.5, SEED, config)
    assert choose_tasks(splits, 0.5, SEED, config) == first
    other = [t["id"] for t in choose_tasks(splits, 0.5, SEED + 1, config)]
    assert other != [t["id"] for t in first]


def test_the_style_fits_the_document(tasks, records, config):
    styles = {s["text"]: s for s in config["styles"]}
    for task in tasks:
        style = styles[task["style"]]
        assert task["kind"] in style["kinds"]
        assert style["foreign"] != ("never" if task["reverse_charge"] else "only")
        if style.get("payment"):
            assert task["payment"] is not None
        if style.get("types"):
            assert records[task["id"]]["facts"]["supplier_type"] in style["types"]
    assert len({t["style"] for t in tasks}) >= 25


def test_vat_rows_add_up_to_the_gross_amounts(tasks):
    for task in tasks:
        if task["reverse_charge"]:
            assert task["vat"] == []
            continue
        rows = {row["rate"]: row for row in task["vat"]}
        gross = sum(Decimal(str(row["net"])) + Decimal(str(row["vat"])) for row in rows.values())
        assert gross == sum(Decimal(str(i["amount"])) for i in task["items"])


def test_the_instructions_follow_the_category(tasks, records):
    seen = Counter()
    for task in tasks:
        category = records[task["id"]]["category"]
        seen[category] += 1
        if category == "ambiguous_purpose":
            assert {i["description"] for i in task["items"]} <= set(task["keep_verbatim"])
            assert "vague" in task["instructions"]
        if category == "prompt_injection":
            assert "aimed at a program" in task["instructions"]
            assert any(k in task["draft"] for k in task["keep_verbatim"])
        if category == "foreign_currency":
            assert task["currency"] != "EUR" and task["currency"] in task["instructions"]
        if category in ("reverse_charge_free_text", None) and task["reverse_charge"]:
            assert any(k in task["draft"] for k in task["keep_verbatim"])
    assert {"ambiguous_purpose", "prompt_injection", "foreign_currency", None} <= set(seen)


def test_the_draft_is_the_only_text_the_checker_rejects_it_for(tasks):
    """Every fact the checker looks for is in the draft, so it never rejects a true text."""
    failures = Counter()
    for task in tasks:
        for reason in check_rewrite(task, task["draft"]):
            if reason != "the text is the draft":
                failures[reason[:30]] += 1
    # An uppercase draft of a vague item list does not hold the names in their own case.
    assert set(failures) <= {"missing text that must stay: '"}, failures
    assert sum(failures.values()) <= 0.01 * len(tasks)


def test_a_rewrite_built_from_the_facts_is_accepted(tasks):
    rejected = [(t["id"], check_rewrite(t, rewrite(t))) for t in tasks]
    assert [r for r in rejected if r[1]] == []


def test_the_pilot_has_every_category(tasks, records):
    pilot = choose_pilot(tasks, records, 40)
    assert len(pilot) == 40
    categories = {records[t["id"]]["category"] for t in pilot}
    assert categories == {records[t["id"]]["category"] for t in tasks}
    assert len({t["id"] for t in pilot}) == 40


def test_a_task_is_made_from_one_record(records, config):
    record = next(r for r in records.values() if r["category"] == "business_private_mixed")
    task = make_task(record, config["styles"], SEED)
    assert task["id"] == record["id"]
    assert "different kinds" in task["instructions"]


def test_the_package_has_everything_and_runs_on_its_own(platform, tmp_path):
    result = rewrite_export.export("tax-booking", SEED, 0.05, 3000, tmp_path / "rewrite-round3")
    folder = tmp_path / "rewrite-round3"
    names = {p.name for p in folder.iterdir()}
    assert names == {
        "README.txt", "RULES.md", "PROMPT_1_PILOT.txt", "PROMPT_2_FULL.txt", "rewrite.py",
        "config.json", "tasks.jsonl", "pilot.jsonl", "out",
    }  # fmt: skip
    assert result["tasks"] == len((folder / "tasks.jsonl").read_text().splitlines())
    with zipfile.ZipFile(result["zip"]) as archive:
        listed = set(archive.namelist())
    assert "rewrite-round3/out/" in listed and "rewrite-round3/rewrite.py" in listed
    assert {f"rewrite-round3/{n}" for n in names if n != "out"} <= listed

    first = subprocess.run(
        [sys.executable, "rewrite.py", "next", "--pilot"],
        cwd=folder, capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert "out/reply-0001.jsonl" in first.stdout
    batch = [
        json.loads(line) for line in (folder / "work" / "current.jsonl").read_text().splitlines()
    ]
    assert 0 < len(batch) <= 40
    reply = "".join(json.dumps({"id": t["id"], "document_text": rewrite(t)}) + "\n" for t in batch)
    (folder / "out" / "reply-0001.jsonl").write_text(reply, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "rewrite.py", "next", "--pilot"],
        cwd=folder, capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert "PILOT DONE" in done.stdout


def test_the_templates_name_the_commands_the_checker_has():
    texts = [
        (rewrite_export.TEMPLATES / name).read_text(encoding="utf-8")
        for name in ("PROMPT_1_PILOT.txt", "PROMPT_2_FULL.txt", "README.txt")
    ]
    for text in texts:
        assert "rewrite.py" in text
    assert "next --pilot" in texts[0] and "pack" in texts[0]
    assert "PILOT DONE" in texts[0] and "DONE" in texts[1]
    rules = (rewrite_export.TEMPLATES / "RULES.md").read_text(encoding="utf-8")
    for field in ("keep_verbatim", "instructions", "draft", "rejected", "document_text"):
        assert field in rules
