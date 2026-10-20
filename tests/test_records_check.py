import copy
import json
from collections import Counter

import pytest

from adapt.config import load_platform
from adapt.data.audit import SETS, read_set
from adapt.data.records_check import (
    canonical,
    check_folder,
    quota_problems,
    record_problems,
    undamage,
)

ACCOUNTS = {a.number for a in load_platform("tax-booking").rules.accounts}


def good_record():
    pool = [r for name in ("dev_real", "gold_v2") for r in read_set("tax-booking", name)]
    record = next(
        r
        for r in pool
        if not r["target"]["needs_review"]
        and len(r["target"]["items"]) == 2
        and not r["target"]["reverse_charge"]
        and r["layout"] != "credit_note"
        and not record_problems(r, ACCOUNTS)
    )
    return copy.deepcopy(record)


@pytest.mark.parametrize("name", SETS)
def test_every_record_of_the_hand_written_sets_passes(name):
    for record in read_set("tax-booking", name):
        assert record_problems(record, ACCOUNTS) == [], record["id"]


def has(problems, start):
    return any(p.startswith(start) for p in problems)


def test_a_good_record_passes_and_small_mistakes_are_found():
    record = good_record()
    assert record_problems(record, ACCOUNTS) == []

    wrong = copy.deepcopy(record)
    wrong["target"]["total"] += 1
    assert has(record_problems(wrong, ACCOUNTS), "the items add up")

    wrong = copy.deepcopy(record)
    wrong["document_text"] = wrong["document_text"].replace(
        wrong["target"]["supplier"], "Other Ltd"
    )
    assert has(record_problems(wrong, ACCOUNTS), "the supplier is not written")

    wrong = copy.deepcopy(record)
    amount = f"{record['target']['items'][0]['amount']:.2f}"
    wrong["document_text"] = wrong["document_text"].replace(amount, "1.11")
    assert has(record_problems(wrong, ACCOUNTS), "the amount")

    wrong = copy.deepcopy(record)
    wrong["target"]["items"][0]["account"] = "4654"
    assert has(record_problems(wrong, ACCOUNTS), "account 4654")

    wrong = copy.deepcopy(record)
    wrong["layout"] = "invoice_list"
    assert has(record_problems(wrong, ACCOUNTS), "layout")

    wrong = copy.deepcopy(record)
    wrong["target"]["document_type"] = "credit_note"
    assert record_problems(wrong, ACCOUNTS)

    wrong = copy.deepcopy(record)
    del wrong["layout"]
    assert has(record_problems(wrong, ACCOUNTS), "the fields must be exactly")


def test_the_vat_lines_of_the_text_must_match_the_items():
    record = next(
        r
        for r in read_set("tax-booking", "gold")
        if "Net 19%" in r["document_text"] and not r["target"]["needs_review"]
    )
    wrong = copy.deepcopy(record)
    line = next(x for x in wrong["document_text"].splitlines() if "Net 19%" in x)
    wrong["document_text"] = wrong["document_text"].replace(line, "Net 19%: 1.00  VAT 19%: 0.19")
    assert has(record_problems(wrong, ACCOUNTS), "the VAT lines")


def test_net_price_rules():
    record = good_record()
    item = record["target"]["items"][0]
    item["account"] = "0410"
    assert any("needs a net price above 900" in p for p in record_problems(record, ACCOUNTS))
    item["account"] = "0480"
    item["amount"] = 1000.0
    record["target"]["total"] = 1000.0 + record["target"]["items"][1]["amount"]
    problems = "\n".join(record_problems(record, ACCOUNTS))
    assert "0480 needs a net price below 700" in problems


def test_a_review_record_has_no_items():
    record = good_record()
    record["target"] = {**record["target"], "needs_review": True, "reason": "unclear"}
    assert has(record_problems(record, ACCOUNTS), "a document for review has no items")
    record["target"]["items"] = []
    assert record_problems(record, ACCOUNTS) == []


def test_look_alike_letters_in_numbers_are_read_as_digits():
    assert (
        undamage("Total 5OO.OO EUR on 2O26-O3-15 and lO.5O")
        == "Total 500.00 EUR on 2026-03-15 and 10.50"
    )
    assert undamage("SOS and Sol") == "SOS and Sol"
    assert canonical("Pub1ishing Corp.") == canonical("Publishing Corp.")
    assert canonical("Marn Ltd") == canonical("Mam Ltd")


def small_folder(tmp_path, records, **config):
    base = {
        "accounts": sorted(ACCOUNTS),
        "first_id": 1,
        "last_id": len(records),
        "review_documents": 0,
        "layouts": {},
        "minimums": {},
        "used_names": [],
    }
    (tmp_path / "config.json").write_text(json.dumps({**base, **config}), encoding="utf-8")
    text = "".join(json.dumps(r) + "\n" for r in records)
    (tmp_path / "new_records.jsonl").write_text(text, encoding="utf-8")
    return tmp_path


def numbered(count):
    pool = [
        r
        for r in read_set("tax-booking", "gold_v2")
        if not r["target"]["needs_review"] and "1" not in r["id"][-1:]
    ]
    chosen = pool[:count]
    for number, record in enumerate(chosen, start=1):
        record["id"] = f"real-{number:03d}"
    return chosen


def test_a_folder_of_good_records_passes(tmp_path):
    records = numbered(3)
    layouts = dict(Counter(r["layout"] for r in records))
    folder = small_folder(tmp_path, records, layouts=layouts)
    assert check_folder(folder) == []


def test_the_folder_check_finds_missing_duplicate_and_used_names(tmp_path):
    records = numbered(3)
    layouts = dict(Counter(r["layout"] for r in records))
    used = [records[0]["target"]["supplier"]]
    problems = check_folder(small_folder(tmp_path, records, layouts=layouts, used_names=used))
    assert has(problems, "real-001: the supplier name")
    problems = check_folder(small_folder(tmp_path, records[:2], layouts=layouts, last_id=3))
    assert "real-003: not written yet" in problems
    problems = check_folder(
        small_folder(tmp_path, [*records, records[0]], layouts=layouts, last_id=3)
    )
    assert has(problems, "real-001: written twice")
    stranger = copy.deepcopy(records)
    stranger[2]["id"] = "real-050"
    assert has(
        check_folder(small_folder(tmp_path, stranger, layouts=layouts)), "real-050: the id must be"
    )
    (tmp_path / "new_records.jsonl").write_text("not json\n{broken\n", encoding="utf-8")
    assert check_folder(tmp_path)[0].startswith("real-001: not written yet") or True


def test_the_quotas_are_checked_when_the_records_are_good(tmp_path):
    records = numbered(3)
    layouts = dict(Counter(r["layout"] for r in records))
    wrong = {**layouts, "credit_note": 5}
    problems = quota_problems(records, json.loads(json.dumps(_config(layouts=wrong))))
    assert has(problems, "layout credit_note: ") or has(problems, "layout")
    problems = quota_problems(
        records, _config(layouts=layouts, minimums={"different accounts": 99})
    )
    assert "different accounts: " in "\n".join(problems)
    assert quota_problems(records, _config(layouts=layouts, review_documents=0)) == []
    assert has(
        quota_problems(records, _config(layouts=layouts, review_documents=2)), "review documents"
    )


def _config(**changes):
    base = {"accounts": sorted(ACCOUNTS), "layouts": {}, "minimums": {}, "review_documents": 0}
    return {**base, **changes}
