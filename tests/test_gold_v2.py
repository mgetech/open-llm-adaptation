"""The second gold file, gold_v2: written after the first round, reported next to gold."""

import json
import shutil

import pytest

from adapt.config import PLATFORMS_DIR
from adapt.data import gold as gold_module
from adapt.data.gold import GOLD_NAMES, gold_path, main
from adapt.eval.gate import GATE_SPLITS, gold_v2_report, load_results
from adapt.eval.runner import (
    ALL_SPLITS,
    DATA_DIR,
    SPLITS,
    default_splits,
    load_documents,
    split_path,
)
from adapt.train.evaluate import evaluate_adapter

RECORD = {
    "id": "gold2-001",
    "platform": "tax-booking",
    "chart": "SKR03",
    "layout": "table_invoice",
    "document_text": "Example GmbH\nPrinter paper 11.90\nTotal: 11.90 EUR",
    "target": {
        "supplier": "Example GmbH",
        "document_date": "2026-01-02",
        "document_type": "invoice",
        "currency": "EUR",
        "total": 11.9,
        "items": [
            {"description": "Printer paper", "amount": 11.9, "account": "4930", "vat_rate": "19"}
        ],
        "reverse_charge": False,
        "needs_review": False,
        "reason": "Office supplies at 19%",
    },
}


def write_gold(root, name, records):
    path = gold_path("tax-booking", root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_there_are_two_gold_files_with_their_own_names():
    assert GOLD_NAMES == ("gold", "gold_v2")
    assert gold_path("tax-booking").name == "gold.jsonl"
    assert gold_path("tax-booking", name="gold_v2").name == "gold_v2.jsonl"
    assert gold_path("tax-booking", name="gold_v2").parent == gold_path("tax-booking").parent


def test_gold_v2_is_a_split_but_not_one_of_the_gate_splits():
    assert "gold_v2" in ALL_SPLITS and "gold_v2" not in SPLITS
    assert "gold_v2" not in GATE_SPLITS


def test_splits_default_to_the_files_that_exist(tmp_path):
    assert default_splits("tax-booking", tmp_path) == SPLITS
    write_gold(tmp_path, "gold_v2", [RECORD])
    assert default_splits("tax-booking", tmp_path) == (*SPLITS, "gold_v2")


def test_gold_v2_documents_are_read_like_gold_documents(tmp_path):
    shutil.copytree(DATA_DIR / "tax-booking", tmp_path / "data" / "tax-booking")
    write_gold(tmp_path / "platforms", "gold_v2", [RECORD])
    data_dir = tmp_path / "data"
    assert split_path("tax-booking", "gold_v2", data_dir).name == "gold_v2.jsonl"

    documents = load_documents("tax-booking", "gold_v2", data_dir)
    assert [d.id for d in documents] == ["gold2-001"]
    document = documents[0]
    assert document.category is None and document.noise is None
    assert document.expected_review is False
    assert document.seen_supplier is False  # the supplier is not one of the training names


def test_an_unknown_split_names_gold_v2_among_the_known_ones():
    with pytest.raises(ValueError, match="gold_v2"):
        load_documents("tax-booking", "train")


def test_an_evaluation_refuses_an_unknown_split_and_names_gold_v2(tmp_path):
    with pytest.raises(ValueError, match="unknown split 'train'.*gold_v2"):
        evaluate_adapter(None, "qwen3-4b", 0, tmp_path, splits=["train"])


def test_the_validator_checks_gold_v2_when_it_exists(tmp_path, monkeypatch, capsys):
    def path_of(platform, root=PLATFORMS_DIR, name="gold"):
        return tmp_path / f"{name}.jsonl"

    def write_v2(records):
        (tmp_path / "gold_v2.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )

    shutil.copy(gold_path("tax-booking"), tmp_path / "gold.jsonl")
    monkeypatch.setattr(gold_module, "gold_path", path_of)

    assert main([]) == 0
    assert "gold_v2" not in capsys.readouterr().out

    write_v2([RECORD])
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "gold.jsonl: " in out and "gold_v2.jsonl: 1 records, all valid" in out

    broken = json.loads(json.dumps(RECORD))
    broken["target"]["items"][0]["account"] = "9999"
    write_v2([broken])
    assert main([]) == 1
    assert "account 9999 is not in the chart" in capsys.readouterr().err


def results_of(accuracy, critical, automation, documents=20):
    metrics = {
        "accuracy_on_automated": {"value": accuracy},
        "critical_error_rate": {"value": critical},
        "automation_rate": {"value": automation},
    }
    return {"metrics": metrics, "documents": [{}] * documents}


def test_the_report_shows_gold_v2_next_to_gold():
    results = {"gold": results_of(0.8, 0.1, 0.9, 80), "gold_v2": results_of(0.95, 0.0, 1.0)}
    shown = dict(gold_v2_report(results))
    assert shown["gold_v2 documents"] == "20"
    assert shown["accuracy on automated documents, gold / gold_v2"] == "0.800 / 0.950"
    assert shown["critical-error rate, gold / gold_v2"] == "0.100 / 0.000"
    assert shown["automation rate, gold / gold_v2"] == "0.900 / 1.000"


def test_the_report_says_when_gold_v2_was_not_run():
    assert gold_v2_report({"gold": results_of(0.8, 0.1, 0.9)}) == [("gold_v2", "not run")]


def test_the_gate_loads_gold_v2_results_when_the_file_exists(tmp_path):
    folder = tmp_path / "tax-booking" / "system"
    folder.mkdir(parents=True)
    (folder / "gold_v2.json").write_text(json.dumps(results_of(1, 0, 1)), encoding="utf-8")
    assert list(load_results(tmp_path, "tax-booking", "system")) == ["gold_v2"]
