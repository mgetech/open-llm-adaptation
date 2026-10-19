import copy
import json
import shutil
from pathlib import Path

import pytest

from adapt.config import load_platform
from adapt.eval.rescore import changes, main, rescore, rescore_row
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
    assert new["manifest"]["rescored"]["checks"] == ["heavy_noise", "vat_mismatch"]
    assert len(new["documents"]) == len(load_documents("tax-booking", "dev", DATA_DIR, 2))
