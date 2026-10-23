"""The checks on gold_v3: size, ids, new supplier names, documents in no other file, and problems
that never show the text of a record."""

import copy
import json

import pytest

from adapt.config import load_platform
from adapt.data.gold import (
    GOLD_V3_RECORDS,
    gold_path,
    gold_v3_problems,
    hide_details,
    main,
    validate_gold,
)

TARGET = {
    "supplier": "placeholder",
    "document_date": "2026-01-02",
    "document_type": "invoice",
    "currency": "EUR",
    "total": 11.9,
    "items": [{"description": "Paper", "amount": 11.9, "account": "4930", "vat_rate": "19"}],
    "reverse_charge": False,
    "needs_review": False,
    "reason": "Office supplies at 19%",
}


def record(number, **changes):
    supplier = f"Quillfeather{number:03d} Holdings"
    target = {**copy.deepcopy(TARGET), "supplier": supplier}
    base = {
        "id": f"gold3-{number:03d}",
        "platform": "tax-booking",
        "chart": "SKR03",
        "layout": "table_invoice",
        "document_text": f"{supplier}\nPaper sheets order {number * 7919}\nTotal: 11.90 EUR",
        "target": target,
    }
    return {**base, **changes}


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture
def folders(tmp_path):
    return tmp_path / "data", tmp_path / "evals"


@pytest.fixture
def gold3(folders):
    _, evals = folders
    return write(evals / "gold_v3.jsonl", [record(n) for n in range(1, GOLD_V3_RECORDS + 1)])


def problems_of(gold3, folders):
    data, evals = folders
    return gold_v3_problems(gold3, "tax-booking", data, evals)


def test_a_file_of_80_new_records_has_no_problem(gold3, folders):
    assert problems_of(gold3, folders) == []


def test_another_number_of_records_is_a_problem(folders, gold3):
    write(gold3, [record(n) for n in range(1, 80)])
    assert problems_of(gold3, folders) == ["gold_v3.jsonl has 79 records, it needs 80"]


def test_an_id_of_another_set_is_a_problem(folders, gold3):
    records = [record(n) for n in range(1, 81)]
    records[4]["id"] = "gold2-005"
    write(gold3, records)
    assert problems_of(gold3, folders) == ["gold2-005: the id must be gold3-NNN"]


def test_a_supplier_in_a_data_file_is_a_problem_that_does_not_name_it(folders, gold3):
    data, _ = folders
    known = record(1)
    write(
        data / "tax-booking" / "round1" / "dev.jsonl", [{**known, "supplier": None, "canary": None}]
    )
    (problem,) = problems_of(gold3, folders)
    assert problem == "gold3-001: the supplier name is also in round1/dev"
    assert known["target"]["supplier"] not in problem


def test_a_supplier_that_differs_only_in_case_and_spacing_is_found(folders, gold3):
    data, _ = folders
    other = record(2)
    other["target"]["supplier"] = "  QUILLFEATHER002   holdings "
    other["document_text"] = "An unrelated document"
    write(
        data / "tax-booking" / "round2" / "train.jsonl",
        [{**other, "supplier": None, "canary": None}],
    )
    assert problems_of(gold3, folders) == ["gold3-002: the supplier name is also in round2/train"]


@pytest.mark.parametrize("other", ["gold", "gold_v2", "dev_real"])
def test_a_supplier_in_another_evaluation_file_is_a_problem(folders, gold3, other):
    _, evals = folders
    existing = record(3)
    existing["id"] = "x-001"
    existing["document_text"] = "A different document entirely"
    write(evals / f"{other}.jsonl", [existing])
    assert problems_of(gold3, folders) == [f"gold3-003: the supplier name is also in {other}"]


def test_a_document_that_is_in_a_training_file_is_a_problem(folders, gold3):
    data, _ = folders
    copied = record(4)
    row = {"id": "train-1", "document_text": copied["document_text"].upper(), "canary": None}
    write(
        data / "tax-booking" / "round3" / "train.jsonl",
        [{**row, "supplier": "Another Name", "target": {"supplier": "Another Name"}}],
    )
    assert problems_of(gold3, folders) == [
        "gold3-004: the document is also in round3/train",
    ]


def test_a_document_that_is_in_another_gold_file_is_a_problem(folders, gold3):
    _, evals = folders
    copied = record(5)
    other = record(5)
    other["id"] = "gold2-001"
    other["target"]["supplier"] = "Some Other Name"
    other["document_text"] = copied["document_text"]
    write(evals / "gold_v2.jsonl", [other])
    assert problems_of(gold3, folders) == ["gold3-005: the document is also in gold_v2"]


def test_the_problems_are_hidden_to_where_they_are_and_how_many():
    problems = [
        "line 3 (gold3-003): the items add up to 11.90, but the total is 12.00",
        "line 3 (gold3-003): account 9999 is not in the chart",
        "gold3-010: the supplier name is also in round1/dev",
        "gold_v3.jsonl has 79 records, it needs 80",
    ]
    assert hide_details(problems) == [
        "line 3 (gold3-003): 2 problem(s)",
        "gold3-010: 1 problem(s)",
        ["gold_v3.jsonl has 79 records, it needs 80"][0] + ": 1 problem(s)",
    ]


def test_a_failing_run_shows_nothing_of_the_record_unless_details_are_asked_for(
    folders, gold3, capsys
):
    records = [record(n) for n in range(1, 81)]
    records[6]["target"]["total"] = 99.9
    write(gold3, records)
    assert main(["--path", str(gold3)]) == 1
    hidden = capsys.readouterr().err
    assert "gold3-007" in hidden and "99.9" not in hidden and "Quillfeather" not in hidden
    assert main(["--path", str(gold3), "--details"]) == 1
    assert "99.9" in capsys.readouterr().err


def test_the_entry_point_accepts_a_file_without_problems(folders, gold3, capsys):
    assert main(["--path", str(gold3)]) == 0
    assert "80 records, all valid" in capsys.readouterr().out


def test_the_committed_gold_v3_passes_every_check_and_shows_nothing_if_it_does_not():
    path = gold_path("tax-booking", name="gold_v3")
    if not path.is_file():
        pytest.skip("gold_v3.jsonl is not in this checkout")
    problems = validate_gold(path, load_platform("tax-booking"))
    problems = problems or gold_v3_problems(path, "tax-booking")
    assert hide_details(problems) == []
