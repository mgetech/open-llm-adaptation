"""The dev_real split: realistic documents written by hand, used to choose a configuration."""

import json
import shutil

import pytest

from adapt.config import load_platform
from adapt.data import gold as gold_module
from adapt.data.gold import (
    DEV_REAL,
    gold_path,
    main,
    train_overlap,
    validate_gold,
)
from adapt.eval.gate import GATE_SPLITS, REPORT_SPLITS, load_results
from adapt.eval.rescore import ALL_SPLITS as RESCORE_SPLITS
from adapt.eval.runner import (
    ALL_SPLITS,
    DATA_DIR,
    SPLITS,
    RunSettings,
    default_splits,
    evaluate,
    load_canaries,
    load_documents,
    resolve_threshold,
    split_path,
    write_results,
)
from fake_model import Model

RECORD = {
    "id": "real-001",
    "platform": "tax-booking",
    "chart": "SKR03",
    "layout": "till_receipt",
    "document_text": "Brandt Bürobedarf\nKopierpapier A4 500 Blatt   5,95\nSumme EUR 5,95",
    "target": {
        "supplier": "Brandt Bürobedarf",
        "document_date": "2026-02-03",
        "document_type": "receipt",
        "currency": "EUR",
        "total": 5.95,
        "items": [
            {"description": "Kopierpapier", "amount": 5.95, "account": "4930", "vat_rate": "19"}
        ],
        "reverse_charge": False,
        "needs_review": False,
        "reason": "Office supplies at 19%",
    },
}


def write_lines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture
def repository(tmp_path):
    """A copy of the current round with a one-document dev_real file, laid out like the project."""
    shutil.copy(DATA_DIR.parent / "results" / "data_manifest.json", _results(tmp_path))
    source = DATA_DIR / "tax-booking" / "round2"
    shutil.copytree(source, tmp_path / "data" / "tax-booking" / "round2")
    write_lines(gold_path("tax-booking", tmp_path / "platforms", DEV_REAL), [RECORD])
    return tmp_path


def _results(root):
    folder = root / "results"
    folder.mkdir(exist_ok=True)
    return folder / "data_manifest.json"


def test_dev_real_is_a_split_for_selection_and_not_for_the_gate_or_a_report():
    assert DEV_REAL in ALL_SPLITS and DEV_REAL in RESCORE_SPLITS
    assert DEV_REAL not in SPLITS
    assert DEV_REAL not in GATE_SPLITS and DEV_REAL not in REPORT_SPLITS


def test_it_is_in_the_folder_of_the_other_hand_written_files():
    assert gold_path("tax-booking", name=DEV_REAL).name == "dev_real.jsonl"
    assert gold_path("tax-booking", name=DEV_REAL).parent == gold_path("tax-booking").parent


def test_splits_default_to_the_files_that_exist(tmp_path):
    assert default_splits("tax-booking", tmp_path) == SPLITS
    write_lines(gold_path("tax-booking", tmp_path, DEV_REAL), [RECORD])
    assert default_splits("tax-booking", tmp_path) == (*SPLITS, DEV_REAL)
    write_lines(gold_path("tax-booking", tmp_path, "gold_v2"), [RECORD])
    assert default_splits("tax-booking", tmp_path) == (*SPLITS, "gold_v2", DEV_REAL)


def test_the_documents_are_read_from_the_platform_folder_in_every_round(repository):
    data_dir = repository / "data"
    assert split_path("tax-booking", DEV_REAL, data_dir).name == "dev_real.jsonl"
    assert split_path("tax-booking", DEV_REAL, data_dir).parent.name == "evals"

    documents = load_documents("tax-booking", DEV_REAL, data_dir)
    assert [d.id for d in documents] == ["real-001"]
    assert documents[0].category is None and documents[0].noise is None
    assert documents[0].expected_review is False
    assert documents[0].seen_supplier is False


def test_the_gate_does_not_load_dev_real_results(tmp_path):
    folder = tmp_path / "tax-booking" / "system"
    folder.mkdir(parents=True)
    (folder / "dev_real.json").write_text("{}", encoding="utf-8")
    assert load_results(tmp_path, "tax-booking", "system") == {}


def test_the_validator_checks_dev_real_when_it_exists(tmp_path, monkeypatch, capsys):
    def path_of(platform, root=None, name="gold"):
        return tmp_path / f"{name}.jsonl"

    shutil.copy(gold_path("tax-booking"), tmp_path / "gold.jsonl")
    monkeypatch.setattr(gold_module, "gold_path", path_of)
    monkeypatch.setattr(gold_module, "train_overlap", lambda *args: [])

    assert main([]) == 0
    assert "dev_real" not in capsys.readouterr().out

    write_lines(tmp_path / "dev_real.jsonl", [RECORD])
    assert main([]) == 0
    assert "dev_real.jsonl: 1 records, all valid" in capsys.readouterr().out

    broken = json.loads(json.dumps(RECORD))
    broken["target"]["items"][0]["account"] = "9999"
    write_lines(tmp_path / "dev_real.jsonl", [broken])
    assert main([]) == 1
    assert "account 9999 is not in the chart" in capsys.readouterr().err


def test_a_document_that_is_in_a_training_file_is_found(tmp_path):
    train = {"document_text": "Some  other text", "canary": None}
    copy = {
        "document_text": "Brandt Bürobedarf\n\nKopierpapier A4 500 blatt 5,95 Summe EUR 5,95",
        "canary": None,
    }
    canary = {"document_text": "x", "canary": {"name": "N", "text_without_name": "y"}}
    write_lines(tmp_path / "data" / "tax-booking" / "round1" / "train.jsonl", [train, canary])
    write_lines(tmp_path / "data" / "tax-booking" / "round2" / "train.jsonl", [train, copy])
    dev_real = write_lines(
        tmp_path / "dev_real.jsonl", [RECORD, {**RECORD, "id": "real-002", "document_text": "y"}]
    )

    problems = train_overlap(dev_real, "tax-booking", tmp_path / "data")
    assert problems == [
        "real-002: the document is also in round1/train",
        "real-001: the document is also in round2/train",
    ]


def test_documents_that_are_not_in_any_training_file_pass(tmp_path):
    write_lines(
        tmp_path / "data" / "tax-booking" / "round1" / "train.jsonl",
        [{"document_text": "Other", "canary": None}],
    )
    dev_real = write_lines(tmp_path / "dev_real.jsonl", [RECORD])
    assert train_overlap(dev_real, "tax-booking", tmp_path / "data") == []


def test_the_validator_reports_a_dev_real_document_found_in_training(tmp_path, monkeypatch, capsys):
    shutil.copy(gold_path("tax-booking"), tmp_path / "gold.jsonl")
    write_lines(tmp_path / "dev_real.jsonl", [RECORD])
    monkeypatch.setattr(
        gold_module, "gold_path", lambda p, r=None, name="gold": tmp_path / f"{name}.jsonl"
    )
    monkeypatch.setattr(gold_module, "train_overlap", lambda *args: ["real-001: in round2/train"])
    assert main([]) == 1
    assert "real-001: in round2/train" in capsys.readouterr().err


def test_no_dev_real_document_is_in_a_training_file_of_the_project():
    path = gold_path("tax-booking", name=DEV_REAL)
    if not path.exists():
        pytest.skip("dev_real.jsonl is not written yet")
    assert train_overlap(path, "tax-booking") == []


def test_the_dev_real_file_of_the_project_is_valid_and_has_30_to_40_documents():
    path = gold_path("tax-booking", name=DEV_REAL)
    if not path.exists():
        pytest.skip("dev_real.jsonl is not written yet")
    assert validate_gold(path, load_platform("tax-booking")) == []
    assert 30 <= len(path.read_text(encoding="utf-8").splitlines()) <= 40


def run_split(server, repository, split, **changes):
    data_dir = repository / "data"
    documents = load_documents("tax-booking", split, data_dir)
    server.handler = Model(documents, load_canaries("tax-booking", data_dir))
    values = dict(
        platform="tax-booking",
        system="demo",
        split=split,
        endpoint=server.url,
        model="tax-booking",
        workers=2,
        data_dir=data_dir,
        out_dir=repository / "out",
    )
    return evaluate(RunSettings(**{**values, **changes}))


def test_the_runner_evaluates_dev_real_with_the_threshold_tuned_on_dev(server, repository):
    server.version = "0.test"
    settings = RunSettings(
        "tax-booking",
        "demo",
        DEV_REAL,
        server.url,
        "tax-booking",
        data_dir=repository / "data",
        out_dir=repository / "out",
    )
    with pytest.raises(ValueError, match="run the dev split first"):
        resolve_threshold(settings)

    dev = run_split(server, repository, "dev")
    write_results(dev, repository / "out" / "tax-booking" / "demo" / "dev.json")
    threshold, source = resolve_threshold(settings)
    assert threshold == dev["manifest"]["threshold"]
    assert source.startswith("tuned on dev")

    results = run_split(server, repository, DEV_REAL, threshold=threshold)
    assert results["split"] == DEV_REAL
    assert [row["id"] for row in results["documents"]] == ["real-001"]
    assert results["manifest"]["data_sha256"]
    assert results["metrics"]["document_accuracy"]["value"] == 1.0
