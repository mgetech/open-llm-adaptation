import copy
import json
import shutil
from pathlib import Path

import pytest

from adapt.config import load_platform
from adapt.eval.configuration import (
    DISAGREE,
    answer_signature,
    combine_results,
    combine_row,
    main,
)
from adapt.eval.metrics import account_groups
from adapt.eval.runner import DATA_DIR, load_documents, results_file

ROUND_TWO = Path(__file__).resolve().parents[1] / "results" / "round2" / "eval"
SMALL, LARGE = "qwen3-1.7b-sft-r2-0", "qwen3-4b-sft-r2-0"


@pytest.fixture(scope="module")
def groups():
    return account_groups(load_platform("tax-booking").rules)


@pytest.fixture(scope="module")
def documents():
    """The first documents of the dev split that should be automated."""
    wanted = [d for d in load_documents("tax-booking", "dev", DATA_DIR, 2) if not d.expected_review]
    return wanted[:6]


def row(document, output=None, *, reasons=(), review=None, by_rule=False, confidence=0.99):
    output = copy.deepcopy(document.target) if output is None else output
    flagged = bool(reasons) if review is None else review
    return {
        "id": document.id,
        "output": output,
        "reasons": list(reasons),
        "needs_review": flagged,
        "by_rule": by_rule,
        "confidence": confidence,
        "item_confidences": None,
    }


def results(rows, system="a", split="dev"):
    return {
        "platform": "tax-booking",
        "system": system,
        "split": split,
        "manifest": {"data_sha256": "x", "data_round": 2, "model": system},
        "metrics": {"canary_leak_rate": {"value": 0.0, "documents": 5}},
        "performance": {"wall_seconds": 10.0},
        "documents": rows,
    }


def pair(documents, groups, first_rows, second_rows, confidence="each"):
    return combine_results(
        results(first_rows, "a"), results(second_rows, "b"), "pair", documents, groups, confidence
    )


def test_two_equal_clean_answers_are_automated_and_correct(documents, groups):
    rows = [row(d) for d in documents]
    combined = pair(documents, groups, rows, rows)
    assert [r["needs_review"] for r in combined["documents"]] == [False] * len(documents)
    assert all(r["correct"] for r in combined["documents"])
    assert combined["metrics"]["automation_rate"]["value"] == 1.0
    assert combined["metrics"]["critical_error_rate"]["value"] == 0.0


def test_answers_that_differ_in_a_vat_rate_go_to_review(documents, groups):
    other = copy.deepcopy(documents[0].target)
    other["items"][0]["vat_rate"] = "7" if other["items"][0]["vat_rate"] != "7" else "19"
    first = [row(documents[0])]
    second = [row(documents[0], other)]
    combined = pair(documents[:1], groups, first, second)
    only = combined["documents"][0]
    assert only["needs_review"] and only["reasons"] == [DISAGREE]
    assert not only["by_rule"]


@pytest.mark.parametrize("field", ["account", "amount"])
def test_a_different_account_or_amount_is_a_disagreement(documents, field):
    other = copy.deepcopy(documents[0].target)
    item = other["items"][0]
    item[field] = "4930" if field == "account" and item[field] != "4930" else item[field]
    if field == "amount":
        item["amount"] = round(item["amount"] + 0.05, 2)
    assert answer_signature(other) != answer_signature(documents[0].target)


def test_a_different_total_or_reverse_charge_flag_is_a_disagreement(documents):
    target = documents[0].target
    assert answer_signature({**target, "total": target["total"] + 1}) != answer_signature(target)
    assert answer_signature({**target, "reverse_charge": not target["reverse_charge"]}) != (
        answer_signature(target)
    )


def test_other_fields_of_the_answer_are_not_compared(documents):
    target = documents[0].target
    changed = {**target, "reason": "another text", "supplier": "Other GmbH"}
    assert answer_signature(changed) == answer_signature(target)


def test_amounts_are_compared_to_the_cent(documents):
    target = copy.deepcopy(documents[0].target)
    close = copy.deepcopy(target)
    close["items"][0]["amount"] = target["items"][0]["amount"] + 0.001
    assert answer_signature(close) == answer_signature(target)


def test_a_document_one_adapter_flagged_goes_to_review_with_the_reasons_of_both(documents, groups):
    first = [row(documents[0], reasons=["model_flagged"])]
    second = [row(documents[0], reasons=["total_mismatch"])]
    only = pair(documents[:1], groups, first, second)["documents"][0]
    assert only["needs_review"]
    assert only["reasons"] == ["model_flagged", "total_mismatch"]


def test_one_adapter_without_an_answer_sends_the_document_to_review(documents, groups):
    first = [row(documents[0])]
    second = [row(documents[0], None, reasons=["invalid_output"], review=True)]
    second[0]["output"] = None
    only = pair(documents[:1], groups, first, second)["documents"][0]
    assert only["needs_review"] and only["reasons"] == ["invalid_output"]


def test_a_low_confidence_counts_with_each_and_not_with_ignore(documents, groups):
    first = [row(documents[0])]
    second = [row(documents[0], reasons=["low_confidence"], confidence=0.4)]
    each = pair(documents[:1], groups, first, second, "each")["documents"][0]
    ignore = pair(documents[:1], groups, first, second, "ignore")["documents"][0]
    assert each["needs_review"] and each["reasons"] == ["low_confidence"]
    assert not ignore["needs_review"] and ignore["reasons"] == []
    assert each["confidence"] == ignore["confidence"] == 0.4


def test_ignoring_the_confidence_still_sends_other_flags_to_review(documents, groups):
    second = [row(documents[0], reasons=["low_confidence", "vat_mismatch"])]
    only = pair(documents[:1], groups, [row(documents[0])], second, "ignore")["documents"][0]
    assert only["needs_review"] and only["reasons"] == ["vat_mismatch"]


def test_a_flag_is_a_rule_flag_only_if_every_flagging_adapter_was_flagged_by_a_rule():
    base = {"output": {"items": []}, "confidence": None}

    def make(review, by_rule, reasons):
        return {**base, "needs_review": review, "by_rule": by_rule, "reasons": reasons}

    both = combine_row(
        make(True, True, ["foreign_currency"]), make(True, True, ["foreign_currency"]), "each"
    )
    mixed = combine_row(
        make(True, True, ["vat_mismatch"]), make(True, False, ["model_flagged"]), "each"
    )
    one = combine_row(make(True, True, ["vat_mismatch"]), make(False, False, []), "each")
    assert both["by_rule"] and one["by_rule"] and not mixed["by_rule"]


def test_the_pair_never_automates_a_document_a_member_sent_to_review(documents, groups):
    wrong = copy.deepcopy(documents[1].target)
    wrong["items"][0]["vat_rate"] = "0" if wrong["items"][0]["vat_rate"] != "0" else "19"
    first = [row(d) for d in documents]
    second = [row(d, wrong if i == 1 else None) for i, d in enumerate(documents)]
    combined = pair(documents, groups, first, second)
    automated = [not r["needs_review"] for r in combined["documents"]]
    assert automated == [i != 1 for i in range(len(documents))]


def test_the_pair_gets_the_worse_canary_result_and_the_added_time(documents, groups):
    rows = [row(d) for d in documents]
    second = results(rows, "b")
    second["metrics"] = {"canary_leak_rate": {"value": 0.2, "documents": 5}}
    second["performance"] = {"wall_seconds": 5.0}
    combined = combine_results(results(rows, "a"), second, "pair", documents, groups)
    assert combined["metrics"]["canary_leak_rate"] == {"value": 0.2, "documents": 5}
    assert combined["performance"]["wall_seconds"] == 15.0
    assert combined["manifest"]["configuration"]["members"] == ["a", "b"]


@pytest.mark.parametrize("what", ["split", "data", "ids", "output"])
def test_results_that_do_not_fit_together_are_refused(documents, groups, what):
    rows = [row(d) for d in documents]
    first, second = results(rows, "a"), results(copy.deepcopy(rows), "b")
    if what == "split":
        second["split"] = "test_ood"
    elif what == "data":
        second["manifest"]["data_sha256"] = "other"
    elif what == "ids":
        second["documents"] = second["documents"][::-1]
    else:
        del second["documents"][0]["output"]
    with pytest.raises(ValueError):
        combine_results(first, second, "pair", documents, groups)


def test_saved_round_2_results_make_a_pair_that_automates_a_subset_of_each_member(tmp_path):
    for system in (SMALL, LARGE):
        source = results_file(ROUND_TWO, "tax-booking", system, "dev")
        target = results_file(tmp_path, "tax-booking", system, "dev")
        target.parent.mkdir(parents=True)
        shutil.copy(source, target)
    args = ["--first", SMALL, "--second", LARGE, "--name", "pair-test"]
    args += ["--splits", "dev", "--round", "2", "--out-dir", str(tmp_path)]
    assert main(args) == 0

    def load(system):
        return json.loads(results_file(tmp_path, "tax-booking", system, "dev").read_text("utf-8"))

    small, large, both = load(SMALL), load(LARGE), load("pair-test")
    automated = {r["id"] for r in both["documents"] if not r["needs_review"]}
    for member in (small, large):
        assert automated <= {r["id"] for r in member["documents"] if not r["needs_review"]}
    assert 0 < len(automated) < len(both["documents"])
    rate = both["metrics"]["critical_error_rate"]["value"]
    assert rate is not None and rate <= large["metrics"]["critical_error_rate"]["value"]


def test_results_without_the_model_output_cannot_be_combined(tmp_path):
    first = results_file(
        ROUND_TWO.parent.parent / "round1" / "eval", "tax-booking", "qwen3-4b-sft-0", "dev"
    )
    assert first.exists()
    system = "qwen3-4b-sft-0"
    target = results_file(tmp_path, "tax-booking", system, "dev")
    target.parent.mkdir(parents=True)
    shutil.copy(first, target)
    with pytest.raises(SystemExit):
        main(
            [
                "--first",
                system,
                "--second",
                system,
                "--name",
                "x",
                "--splits",
                "dev",
                "--round",
                "1",
                "--out-dir",
                str(tmp_path),
            ]
        )
