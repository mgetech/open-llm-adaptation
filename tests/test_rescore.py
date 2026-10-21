import copy
import json
import shutil
from pathlib import Path

import pytest

from adapt.config import load_platform
from adapt.eval.rescore import (
    changes,
    format_effects,
    main,
    rescore,
    rescore_row,
    rows_without_limit,
    rule_effects,
    tune_on_dev,
)
from adapt.eval.runner import DATA_DIR, load_documents, results_file
from adapt.eval.scoring import EvalDocument

ROUND_TWO = Path(__file__).resolve().parents[1] / "results" / "round2" / "eval"
SYSTEM = "qwen3-1.7b-sft-r2-0"
TEXT = (
    "Nordpaper GmbH  Invoice 2026-1184\n"
    "Printer paper A4 ... 23.80\n"
    "Tax handbook 2026 ... 21.40\n"
    "Net 19%: 20.00  VAT 19%: 3.80\n"
    "Net 7%: 20.00  VAT 7%: 1.40\n"
    "Total EUR 45.20"
)
TARGET = {
    "supplier": "Nordpaper GmbH",
    "currency": "EUR",
    "total": 45.2,
    "items": [
        {"description": "Printer paper A4", "amount": 23.8, "account": "4930", "vat_rate": "19"},
        {"description": "Tax handbook 2026", "amount": 21.4, "account": "4940", "vat_rate": "7"},
    ],
    "reverse_charge": False,
    "needs_review": False,
}


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


def document(text=TEXT):
    return EvalDocument("d1", text, "SKR03", TARGET, False)


def row(output=None, *, reasons=(), review=False):
    return {
        "id": "d1",
        "expected_review": False,
        "needs_review": review,
        "correct": not review,
        "answer_correct": True,
        "severity": None,
        "reasons": list(reasons),
        "confidence": 0.99,
        "item_confidences": [0.99, 0.99],
        "output": copy.deepcopy(TARGET) if output is None else output,
        "by_rule": False,
    }


def wrong_rate():
    output = copy.deepcopy(TARGET)
    output["items"][0]["vat_rate"] = "7"
    return output


def test_a_right_answer_is_left_as_it_was(platform):
    assert rescore_row(row(), document(), platform) == row()


def test_an_answer_that_differs_from_the_printed_vat_goes_to_review_by_rule(platform):
    new = rescore_row(row(wrong_rate()), document(), platform)
    assert new["needs_review"] and new["reasons"] == ["vat_mismatch"] and new["by_rule"]
    assert new["output"] == wrong_rate()


def test_the_vat_flag_goes_before_the_confidence_reasons(platform):
    saved = row(wrong_rate(), reasons=["low_confidence"], review=True)
    assert rescore_row(saved, document(), platform)["reasons"] == ["vat_mismatch", "low_confidence"]


def test_a_flag_of_the_model_keeps_the_document_a_model_flag(platform):
    saved = row(wrong_rate(), reasons=["model_flagged"], review=True)
    new = rescore_row(saved, document(), platform)
    assert new["reasons"] == ["model_flagged", "vat_mismatch"] and not new["by_rule"]


def test_a_document_with_an_unreadable_amount_is_stopped_before_the_model(platform):
    text = TEXT.replace("Printer paper A4 ... 23.80", "Printer paper A4 ... 2?.80")
    new = rescore_row(row(), document(text), platform)
    assert new["needs_review"] and new["reasons"] == ["heavy_noise"] and new["by_rule"]
    assert new["output"] is None and new["confidence"] is None


def test_the_changes_say_what_was_caught_and_what_was_flagged_wrongly(platform):
    wrong = row(wrong_rate())
    wrong["answer_correct"] = False
    cases = [(row(), row()), (wrong, rescore_row(wrong, document(), platform))]
    flagged_right = row()
    flagged_after = {**flagged_right, "needs_review": True, "reasons": ["vat_mismatch"]}
    cases.append((flagged_right, flagged_after))
    counted = changes([c[0] for c in cases], [c[1] for c in cases])
    assert counted == {"vat_mismatch caught": 1, "vat_mismatch false": 1}


def test_a_document_that_was_in_review_is_not_counted(platform):
    before = row(reasons=["model_flagged"], review=True)
    after = {**before, "reasons": ["model_flagged", "vat_mismatch"]}
    assert not changes([before], [after])


def test_results_without_the_model_output_cannot_be_scored_again(platform):
    saved = {"system": "x", "documents": [{"id": "d1"}], "metrics": {}, "manifest": {}}
    with pytest.raises(ValueError, match="model output"):
        rescore(saved, [document()], platform)


def test_documents_that_do_not_match_the_results_are_refused(platform):
    saved = {"system": "x", "documents": [row()], "metrics": {}, "manifest": {}}
    with pytest.raises(ValueError, match="do not match"):
        rescore(saved, [EvalDocument("other", TEXT, "SKR03", TARGET, False)], platform)


def test_saved_round_2_results_can_only_lose_automated_documents(tmp_path):
    source = results_file(ROUND_TWO, "tax-booking", SYSTEM, "dev")
    copied = results_file(tmp_path / "in", "tax-booking", SYSTEM, "dev")
    copied.parent.mkdir(parents=True)
    shutil.copy(source, copied)
    args = ["--systems", SYSTEM, "--splits", "dev", "--round", "2"]
    args += ["--in-dir", str(tmp_path / "in"), "--out-dir", str(tmp_path / "out")]
    assert main(args) == 0

    def load(folder):
        return json.loads(
            results_file(tmp_path / folder, "tax-booking", SYSTEM, "dev").read_text("utf-8")
        )

    old, new = load("in"), load("out")
    was = {r["id"] for r in old["documents"] if not r["needs_review"]}
    now = {r["id"] for r in new["documents"] if not r["needs_review"]}
    assert now <= was and now != was
    assert new["manifest"]["rescored"]["checks"] == [
        "heavy_noise",
        "vat_mismatch",
        "mixed_private",
        "reverse_charge_cue",
    ]
    assert len(new["documents"]) == len(load_documents("tax-booking", "dev", DATA_DIR, 2))


def mixed_answer():
    output = copy.deepcopy(TARGET)
    output["items"][1]["account"] = "1800"
    return output


def test_an_answer_with_a_private_and_a_business_account_goes_to_review_by_rule(platform):
    new = rescore_row(row(mixed_answer()), document(), platform)
    assert new["needs_review"] and new["reasons"] == ["mixed_private"] and new["by_rule"]


def test_only_the_rules_asked_for_are_applied(platform):
    saved = row(mixed_answer())
    assert rescore_row(saved, document(), platform, ("vat_mismatch",)) == saved
    new = rescore_row(saved, document(), platform, ("mixed_private",))
    assert new["reasons"] == ["mixed_private"]


def test_a_reason_the_row_already_has_is_not_added_again(platform):
    saved = row(mixed_answer(), reasons=["mixed_private"], review=True)
    assert rescore_row(saved, document(), platform) == saved


def test_the_effect_of_each_new_rule_is_shown_alone_and_together(platform):
    mixed = row(mixed_answer())
    mixed["answer_correct"] = False
    mixed["correct"] = False
    wanted = EvalDocument("d1", TEXT, "SKR03", TARGET, True)
    results = {"system": "s", "documents": [mixed]}
    lines = rule_effects(results, [wanted], platform)
    assert [line["label"] for line in lines] == [
        "without them",
        "mixed_private",
        "reverse_charge_cue",
        "both",
    ]
    by_label = {line["label"]: line for line in lines}
    assert by_label["without them"]["review_recall"] == 0.0
    assert by_label["mixed_private"]["review_recall"] == 1.0
    assert by_label["mixed_private"]["caught"] == 1 and by_label["mixed_private"]["false"] == 0
    assert by_label["reverse_charge_cue"]["review_recall"] == 0.0
    assert by_label["both"]["review_recall"] == 1.0
    assert "mixed_private" in format_effects("s", "dev", 1, lines)


ROUND_THREE = Path(__file__).resolve().parents[1] / "results" / "round3" / "eval"
SYSTEM_THREE = "qwen3-4b-sft-r3-0"


def asset_account():
    output = copy.deepcopy(TARGET)
    output["items"][0]["account"] = "0480"  # an asset account for an expense: critical
    return output


def dev_small():
    """d1 and d3 are right, d2 has a critical error and a lower confidence than d1 and d3."""
    documents = [EvalDocument(f"d{i}", TEXT, "SKR03", TARGET, False) for i in (1, 2, 3)]
    outputs = [None, asset_account(), None]
    confidences = [0.99, 0.90, 0.95]
    rows = [
        {**row(output), "id": f"d{i}", "confidence": confidence}
        for i, output, confidence in zip((1, 2, 3), outputs, confidences, strict=True)
    ]
    rows[1]["answer_correct"] = False
    rows[1]["correct"] = False
    saved = {
        "system": "x",
        "documents": rows,
        "metrics": {"canary_leak_rate": {"value": 0.0, "documents": 0}},
        "manifest": {"threshold": 0.0},
        "confidence_threshold": {"value": 0.0, "source": "given"},
    }
    return saved, documents


def test_a_document_that_only_the_confidence_limit_stopped_gets_its_severity_back(platform):
    saved, documents = dev_small()
    saved["documents"][1].update(needs_review=True, reasons=["low_confidence"])
    rows = rows_without_limit(saved, documents, platform)
    assert [r["needs_review"] for r in rows] == [False, False, False]
    assert [r["severity"] for r in rows] == [None, "critical", None]


def test_the_threshold_is_chosen_again_so_that_the_critical_document_is_left_out(platform):
    saved, documents = dev_small()
    tuned = tune_on_dev(saved, documents, platform)
    assert tuned.threshold == 0.95 and tuned.critical_reached


def test_the_chosen_threshold_is_used_on_the_results_that_are_scored_again(platform):
    saved, documents = dev_small()
    tuned = tune_on_dev(saved, documents, platform)
    dev, _ = rescore(saved, documents, platform, tuned, "dev")
    assert dev["manifest"]["threshold"] == 0.95
    assert [r["needs_review"] for r in dev["documents"]] == [False, True, False]
    assert dev["documents"][1]["reasons"] == ["low_confidence"]
    assert dev["metrics"]["critical_error_rate"]["value"] == 0.0
    assert dev["confidence_threshold"]["source"] == "tuned again on dev"
    assert dev["confidence_threshold"]["curve"]
    other, _ = rescore(saved, documents, platform, tuned, "dev_real")
    assert other["manifest"]["threshold"] == 0.95 and "curve" not in other["confidence_threshold"]


def test_without_a_new_threshold_the_one_of_the_run_stays(platform):
    saved, documents = dev_small()
    kept, _ = rescore(saved, documents, platform)
    assert kept["confidence_threshold"] == saved["confidence_threshold"]
    assert kept["manifest"]["threshold"] == 0.0


def test_the_command_chooses_the_threshold_on_dev_and_uses_it_on_dev_real(tmp_path):
    for split in ("dev", "dev_real"):
        copied = results_file(tmp_path / "in", "tax-booking", SYSTEM_THREE, split)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(results_file(ROUND_THREE, "tax-booking", SYSTEM_THREE, split), copied)
    arguments = ["--systems", SYSTEM_THREE, "--splits", "dev", "dev_real", "--round", "3"]
    out = tmp_path / "out"
    assert (
        main([*arguments, "--in-dir", str(tmp_path / "in"), "--out-dir", str(out), "--retune"]) == 0
    )
    dev, dev_real = (
        json.loads(results_file(out, "tax-booking", SYSTEM_THREE, s).read_text(encoding="utf-8"))
        for s in ("dev", "dev_real")
    )
    assert dev["manifest"]["threshold"] == dev_real["manifest"]["threshold"] > 0.9
    assert dev["metrics"]["critical_error_rate"]["value"] <= 0.005
    assert dev["metrics"]["accuracy_on_automated"]["value"] >= 0.95
    assert dev["metrics"]["automation_rate"]["value"] >= 0.6


def test_retuning_needs_the_saved_dev_results(tmp_path):
    with pytest.raises(SystemExit, match="needs the saved dev results"):
        main(["--systems", SYSTEM_THREE, "--round", "3", "--in-dir", str(tmp_path), "--retune"])
