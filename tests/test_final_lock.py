"""The lock on the final claim: test_ood and the gold files back one claim."""

import json
import subprocess

import pytest

from adapt.eval import final_lock
from adapt.eval.final_lock import FinalLockError, claim, lock_file, read_claims
from adapt.eval.runner import RESULTS_ROOT, claim_final_split, main
from adapt.models import load_registry
from adapt.train import evaluate as train_evaluate
from adapt.train.evaluate import evaluate_adapter

FILES = {"test_ood_sha256": "aaa", "gold_sha256": {"gold": "g1", "gold_v2": "g2"}}


def test_the_first_claim_is_written_and_the_same_system_may_go_on(tmp_path):
    path = tmp_path / "final.lock.json"
    claim(path, FILES, "demo", "adapter-1", 3)
    claim(path, FILES, "demo", "adapter-1", 3)
    claims = read_claims(path)
    assert len(claims) == 1
    assert claims[0] == {
        **FILES,
        "system": "demo",
        "adapter_hash": "adapter-1",
        "round": 3,
        "override": False,
    }


def test_another_system_on_the_same_files_is_refused(tmp_path):
    path = tmp_path / "final.lock.json"
    claim(path, FILES, "demo", "adapter-1", 3)
    with pytest.raises(FinalLockError, match="demo already made the final claim"):
        claim(path, FILES, "other", "adapter-2", 3)
    assert len(read_claims(path)) == 1


def test_the_same_system_with_another_adapter_is_refused(tmp_path):
    path = tmp_path / "final.lock.json"
    claim(path, FILES, "demo", "adapter-1", 3)
    with pytest.raises(FinalLockError):
        claim(path, FILES, "demo", "adapter-2", 3)


def test_the_override_runs_and_is_written_to_the_lock(tmp_path):
    path = tmp_path / "final.lock.json"
    claim(path, FILES, "demo", "adapter-1", 3)
    claim(path, FILES, "other", "adapter-2", 3, override=True)
    claims = read_claims(path)
    assert [c["system"] for c in claims] == ["demo", "other"]
    assert [c["override"] for c in claims] == [False, True]


def test_new_files_open_a_new_claim(tmp_path):
    path = tmp_path / "final.lock.json"
    claim(path, FILES, "demo", "adapter-1", 3)
    claim(path, {**FILES, "test_ood_sha256": "bbb"}, "other", "adapter-2", 4)
    claim(path, {**FILES, "gold_sha256": {"gold": "g1", "gold_v3": "g3"}}, "third", None, 4)
    assert len(read_claims(path)) == 3
    assert [c["override"] for c in read_claims(path)] == [False, False, False]


def test_only_results_inside_the_repository_results_folder_have_a_lock(tmp_path):
    assert lock_file(tmp_path / "scratch", tmp_path / "results") is None
    assert lock_file(tmp_path / "results" / "round3" / "eval", tmp_path / "results") == (
        tmp_path / "results" / final_lock.LOCK_NAME
    )


def run(tmp_path, system, split="test_ood", **changes):
    values = dict(
        out_dir=tmp_path / "results" / "round3" / "eval",
        data_round=3,
        results_root=tmp_path / "results",
    )
    claim_final_split("tax-booking", system, split, **{**values, **changes})


def test_a_run_on_test_ood_makes_the_claim_with_the_hashes_of_the_files(tmp_path):
    run(tmp_path, "demo")
    [made] = read_claims(tmp_path / "results" / "final.lock.json")
    assert made["system"] == "demo" and made["round"] == 3
    assert len(made["test_ood_sha256"]) == 64
    assert set(made["gold_sha256"]) == {"gold", "gold_v2"}


def test_a_second_system_on_test_ood_or_a_gold_set_is_refused(tmp_path):
    run(tmp_path, "demo", "test_ood")
    for split in ("test_ood", "gold", "gold_v2"):
        with pytest.raises(ValueError, match="already made the final claim"):
            run(tmp_path, "other", split)
    run(tmp_path, "other", "gold", override=True)


def test_other_splits_and_the_baseline_make_no_claim(tmp_path):
    for split in ("dev", "dev_real", "test_iid", "edge"):
        run(tmp_path, "demo", split)
    run(tmp_path, "demo-fewshot", "test_ood", few_shot=True)
    assert not (tmp_path / "results" / "final.lock.json").exists()


def test_a_run_outside_the_results_folder_makes_no_claim(tmp_path):
    run(tmp_path, "demo", out_dir=tmp_path / "scratch")
    assert not (tmp_path / "results" / "final.lock.json").exists()


def test_the_command_line_is_refused_before_the_model_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "adapt.eval.runner.claim_final_split",
        lambda *args, **kwargs: (_ for _ in ()).throw(FinalLockError("locked")),
    )

    def no_request(*args, **kwargs):
        raise AssertionError("the model was used")

    monkeypatch.setattr("adapt.eval.runner.evaluate", no_request)
    with pytest.raises(SystemExit, match="error: locked"):
        main(["--system", "demo", "--split", "test_ood", "--out-dir", str(tmp_path)])


def test_the_adapter_evaluation_is_refused_before_any_server_starts(tmp_path, monkeypatch):
    folder = tmp_path / "tax-booking" / "qwen3-4b-r3-0"
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    asked = []

    def refuse(platform, system, split, out_dir, **kwargs):
        asked.append((system, split, kwargs["override"]))
        if split == "test_ood":
            raise FinalLockError("locked")

    def no_process(*args, **kwargs):
        raise AssertionError("a server was started")

    monkeypatch.setattr(train_evaluate, "claim_final_split", refuse)
    monkeypatch.setattr(subprocess, "Popen", no_process)
    with pytest.raises(ValueError, match="locked"):
        evaluate_adapter(
            load_registry(),
            "qwen3-4b",
            0,
            tmp_path / "eval",
            adapters_dir=tmp_path,
            splits=["dev", "test_ood"],
            data_round=3,
            override_final_lock=True,
        )
    assert asked == [("qwen3-4b-sft-r3-0", "dev", True), ("qwen3-4b-sft-r3-0", "test_ood", True)]


def test_the_committed_lock_names_the_round_3_claim():
    claims = json.loads((RESULTS_ROOT / "final.lock.json").read_text(encoding="utf-8"))["claims"]
    assert [(c["system"], c["round"]) for c in claims] == [("qwen3-4b-sft-r3-0", 3)]
