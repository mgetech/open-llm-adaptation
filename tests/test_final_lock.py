"""The lock on the final claim: declared before the run, with the configurations and the
shipped one fixed."""

import json
import subprocess

import pytest

from adapt.eval import final_lock
from adapt.eval.declare_claim import main as declare_main
from adapt.eval.declare_claim import parse_configuration, seal_file, sealed_lines, status
from adapt.eval.final_lock import (
    FinalLockError,
    declare,
    lock_file,
    read_claims,
    read_sealed,
    seal,
    start_run,
)
from adapt.eval.runner import RESULTS_ROOT, claim_final_split, final_files, main
from adapt.models import load_registry
from adapt.train import evaluate as train_evaluate
from adapt.train.evaluate import evaluate_adapter

FILES = {
    "test_iid_sha256": "iii",
    "test_ood_sha256": "aaa",
    "edge_sha256": "eee",
    "gold_sha256": {"gold": "g1", "gold_v2": "g2"},
}
FRESH = {
    "test_iid_sha256": "jjj",
    "test_ood_sha256": "bbb",
    "edge_sha256": "fff",
    "gold_sha256": {"gold": "g1", "gold_v2": "g2", "gold_v3": "g3"},
}
SINGLE = {"demo": [{"system": "demo", "adapter_hash": "h1"}]}
PAIR = {
    "demo": [{"system": "demo", "adapter_hash": "h1"}],
    "agree": [
        {"system": "first", "adapter_hash": "h2"},
        {"system": "second", "adapter_hash": "h3"},
    ],
}


def test_a_claim_is_declared_with_its_configurations_and_the_shipped_one(tmp_path):
    path = tmp_path / "final.lock.json"
    declare(path, FILES, 3, PAIR, "agree")
    [claim] = read_claims(path)
    assert claim["shipped"] == "agree" and claim["started"] is False and claim["runs"] == []
    assert [c["name"] for c in claim["configurations"]] == ["demo", "agree"]
    assert [a["system"] for a in claim["configurations"][1]["adapters"]] == ["first", "second"]


def test_a_declaration_needs_valid_configurations(tmp_path):
    path = tmp_path / "final.lock.json"
    with pytest.raises(FinalLockError, match="at least one"):
        declare(path, FILES, 3, {}, "demo")
    with pytest.raises(FinalLockError, match="not in the list"):
        declare(path, FILES, 3, SINGLE, "other")
    three = [{"system": s, "adapter_hash": "h"} for s in "abc"]
    with pytest.raises(FinalLockError, match="one or two adapters"):
        declare(path, FILES, 3, {"x": three}, "x")
    twice = [{"system": "a", "adapter_hash": "h"}] * 2
    with pytest.raises(FinalLockError, match="same system twice"):
        declare(path, FILES, 3, {"x": twice}, "x")
    with pytest.raises(FinalLockError, match="without a hash"):
        declare(path, FILES, 3, {"x": [{"system": "a", "adapter_hash": ""}]}, "x")
    assert not path.exists()


def test_a_run_without_a_declaration_is_refused(tmp_path):
    with pytest.raises(FinalLockError, match="no claim is declared"):
        start_run(tmp_path / "final.lock.json", FILES, "demo", "h1")
    with pytest.raises(FinalLockError, match="no claim is declared"):
        start_run(tmp_path / "final.lock.json", FILES, "demo", "h1", override=True)


def test_every_adapter_of_a_declared_configuration_may_run_and_starts_the_claim(tmp_path):
    path = tmp_path / "final.lock.json"
    declare(path, FILES, 3, PAIR, "agree")
    for system, adapter_hash in [("demo", "h1"), ("first", "h2"), ("second", "h3")]:
        start_run(path, FILES, system, adapter_hash)
    start_run(path, FILES, "demo", "h1")
    [claim] = read_claims(path)
    assert claim["started"] is True
    assert [r["system"] for r in claim["runs"]] == ["demo", "first", "second"]
    assert not any(r["override"] for r in claim["runs"])


def test_a_system_or_adapter_that_is_not_in_the_list_is_refused(tmp_path):
    path = tmp_path / "final.lock.json"
    declare(path, FILES, 3, SINGLE, "demo")
    with pytest.raises(FinalLockError, match="other is not in the claim"):
        start_run(path, FILES, "other", "h9")
    with pytest.raises(FinalLockError, match="demo is not in the claim"):
        start_run(path, FILES, "demo", "changed-weights")
    assert read_claims(path)[0]["started"] is False


def test_the_override_runs_an_outsider_and_marks_it_without_changing_the_claim(tmp_path):
    path = tmp_path / "final.lock.json"
    declare(path, FILES, 3, SINGLE, "demo")
    start_run(path, FILES, "other", "h9", override=True)
    [claim] = read_claims(path)
    assert claim["shipped"] == "demo"
    assert claim["runs"] == [{"system": "other", "adapter_hash": "h9", "override": True}]


def test_a_claim_that_has_not_started_can_be_declared_again(tmp_path):
    path = tmp_path / "final.lock.json"
    declare(path, FILES, 3, SINGLE, "demo")
    declare(path, FILES, 3, PAIR, "agree")
    [claim] = read_claims(path)
    assert claim["shipped"] == "agree"


def test_a_claim_that_has_started_cannot_change(tmp_path):
    path = tmp_path / "final.lock.json"
    declare(path, FILES, 3, PAIR, "demo")
    start_run(path, FILES, "demo", "h1")
    with pytest.raises(FinalLockError, match="has started"):
        declare(path, FILES, 3, PAIR, "agree")
    assert read_claims(path)[0]["shipped"] == "demo"


def test_new_files_need_their_own_declaration(tmp_path):
    path = tmp_path / "final.lock.json"
    declare(path, FILES, 3, SINGLE, "demo")
    start_run(path, FILES, "demo", "h1")
    with pytest.raises(FinalLockError, match="no claim is declared"):
        start_run(path, FRESH, "demo", "h1")
    declare(path, FRESH, 4, SINGLE, "demo")
    start_run(path, FRESH, "demo", "h1")
    assert len(read_claims(path)) == 2


@pytest.mark.parametrize("key", ["test_iid_sha256", "test_ood_sha256", "edge_sha256"])
def test_a_test_file_that_backs_a_started_claim_cannot_back_another(tmp_path, key):
    path = tmp_path / "final.lock.json"
    declare(path, FILES, 3, SINGLE, "demo")
    start_run(path, FILES, "demo", "h1")
    reused = {**FRESH, key: FILES[key]}
    with pytest.raises(FinalLockError, match=f"{key.removesuffix('_sha256')} already backs"):
        declare(path, reused, 4, SINGLE, "demo")
    assert len(read_claims(path)) == 1


def test_a_new_claim_needs_a_gold_file_that_no_earlier_claim_used(tmp_path):
    path = tmp_path / "final.lock.json"
    declare(path, FILES, 3, SINGLE, "demo")
    start_run(path, FILES, "demo", "h1")
    same_gold = {**FRESH, "gold_sha256": FILES["gold_sha256"]}
    with pytest.raises(FinalLockError, match="no gold file is new"):
        declare(path, same_gold, 4, SINGLE, "demo")
    declare(path, FRESH, 4, SINGLE, "demo")


def test_a_claim_that_has_not_started_does_not_block_a_new_declaration(tmp_path):
    path = tmp_path / "final.lock.json"
    declare(path, FILES, 3, SINGLE, "demo")
    declare(path, {**FRESH, "test_ood_sha256": FILES["test_ood_sha256"]}, 4, SINGLE, "demo")
    assert len(read_claims(path)) == 2


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


def declare_here(tmp_path, adapter_hash="h1"):
    config = {"demo": [{"system": "demo", "adapter_hash": adapter_hash}]}
    files = final_files("tax-booking", data_round=3)
    declare(tmp_path / "results" / "final.lock.json", files, 3, config, "demo")


def test_a_run_on_the_files_of_the_round_follows_the_declaration(tmp_path, monkeypatch):
    monkeypatch.setattr("adapt.eval.runner.hash_directory", lambda path, names=None: "h1")
    with pytest.raises(ValueError, match="no claim is declared"):
        run(tmp_path, "demo", adapter_path=tmp_path)
    declare_here(tmp_path)
    for split in ("test_iid", "test_ood", "edge", "gold", "gold_v2"):
        run(tmp_path, "demo", split, adapter_path=tmp_path)
        with pytest.raises(ValueError, match="other is not in the claim"):
            run(tmp_path, "other", split, adapter_path=tmp_path)
    run(tmp_path, "other", "gold", adapter_path=tmp_path, override=True)


def test_other_splits_and_the_baseline_need_no_declaration(tmp_path):
    for split in ("dev", "dev_real"):
        run(tmp_path, "demo", split)
    run(tmp_path, "demo-fewshot", "test_ood", few_shot=True)
    assert not (tmp_path / "results" / "final.lock.json").exists()


def test_a_run_outside_the_results_folder_needs_no_declaration(tmp_path):
    run(tmp_path, "demo", out_dir=tmp_path / "scratch")


def test_the_command_line_is_refused_before_the_model_is_used(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise FinalLockError("locked")

    def no_request(*args, **kwargs):
        raise AssertionError("the model was used")

    monkeypatch.setattr("adapt.eval.runner.claim_final_split", refuse)
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


def make_adapter(folder):
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    return folder


def test_a_configuration_is_read_from_its_text_with_the_hashes_of_the_folders(tmp_path):
    one = make_adapter(tmp_path / "one")
    two = make_adapter(tmp_path / "two")
    (two / "adapter_model.safetensors").write_bytes(b"other weights")
    name, adapters = parse_configuration(f"agree=first:{one}+second:{two}")
    assert name == "agree"
    assert [a["system"] for a in adapters] == ["first", "second"]
    assert adapters[0]["adapter_hash"] != adapters[1]["adapter_hash"]
    assert len(adapters[0]["adapter_hash"]) == 64


def test_other_files_in_the_adapter_folder_do_not_change_its_hash(tmp_path):
    plain = make_adapter(tmp_path / "plain")
    extra = make_adapter(tmp_path / "extra")
    (extra / "README.md").write_text("notes", encoding="utf-8")
    (extra / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert parse_configuration(f"a=s:{plain}")[1] == parse_configuration(f"a=s:{extra}")[1]
    (extra / "adapter_model.safetensors").write_bytes(b"changed weights")
    assert parse_configuration(f"a=s:{plain}")[1] != parse_configuration(f"a=s:{extra}")[1]


@pytest.mark.parametrize("text", ["no-equals", "name=", "name=nofolder", "name=a:/no/folder"])
def test_a_configuration_with_a_wrong_text_or_no_adapter_is_refused(text):
    with pytest.raises(ValueError):
        parse_configuration(text)


def test_the_status_shows_the_claim_and_the_results_that_are_missing(tmp_path):
    lock = tmp_path / "final.lock.json"
    declare(lock, final_files("tax-booking", data_round=3), 3, PAIR, "agree")
    text = status("tax-booking", 3, lock)
    assert "shipped: agree" in text and "started: False" in text
    assert "first: results for none; missing test_iid, test_ood, edge, gold, gold_v2" in text
    assert status("tax-booking", 3, tmp_path / "other.json").startswith("no claim is declared")


def test_the_command_refuses_a_missing_adapter_and_a_missing_shipped_name():
    with pytest.raises(SystemExit, match="no adapter in"):
        declare_main(["--config", "a=a:/no/such/folder", "--shipped", "a"])
    with pytest.raises(SystemExit):
        declare_main(["--config", "a=a:/x"])


def test_the_committed_lock_holds_the_round_3_claim_as_started():
    claims = json.loads((RESULTS_ROOT / "final.lock.json").read_text(encoding="utf-8"))["claims"]
    [claim] = claims
    assert claim["round"] == 3 and claim["started"] is True
    assert claim["shipped"] == "qwen3-4b-sft-r3-0"
    assert claim["test_iid_sha256"] and claim["test_ood_sha256"] and claim["edge_sha256"]
    assert set(claim["gold_sha256"]) == {"gold", "gold_v2"}
    assert [r["system"] for r in claim["runs"]] == ["qwen3-4b-sft-r3-0"]


def test_a_file_is_sealed_with_its_hash_and_the_date_next_to_the_claims(tmp_path):
    path = tmp_path / "final.lock.json"
    declare(path, FILES, 3, SINGLE, "demo")
    assert seal(path, "gold_v3", "evals/gold_v3.jsonl", "abc", "2026-10-22") is True
    assert read_sealed(path) == {
        "gold_v3": {"file": "evals/gold_v3.jsonl", "sha256": "abc", "sealed": "2026-10-22"}
    }
    assert len(read_claims(path)) == 1
    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["claims", "sealed"]


def test_sealing_the_same_hash_again_changes_nothing_and_another_hash_is_refused(tmp_path):
    path = tmp_path / "final.lock.json"
    seal(path, "gold_v3", "evals/gold_v3.jsonl", "abc", "2026-10-22")
    before = path.read_text(encoding="utf-8")
    assert seal(path, "gold_v3", "evals/gold_v3.jsonl", "abc", "2026-10-30") is False
    assert path.read_text(encoding="utf-8") == before
    with pytest.raises(FinalLockError, match="sealed on 2026-10-22 with another hash"):
        seal(path, "gold_v3", "evals/gold_v3.jsonl", "changed", "2026-10-30")


def test_claims_written_later_keep_the_sealed_files(tmp_path):
    path = tmp_path / "final.lock.json"
    seal(path, "gold_v3", "evals/gold_v3.jsonl", "abc", "2026-10-22")
    declare(path, FILES, 3, SINGLE, "demo")
    start_run(path, FILES, "demo", "h1")
    assert read_sealed(path)["gold_v3"]["sha256"] == "abc"
    assert read_claims(path)[0]["started"] is True


def test_the_command_seals_a_file_by_its_hash_and_says_if_it_still_matches(tmp_path):
    lock = tmp_path / "final.lock.json"
    target = tmp_path / "gold_v9.jsonl"
    target.write_text("one line", encoding="utf-8")
    first = seal_file(target, lock, today="2026-10-22")
    assert first.startswith("sealed: gold_v9 ") and first.endswith("on 2026-10-22")
    assert seal_file(target, lock, today="2026-11-01").startswith("already sealed: gold_v9 ")
    target.write_text("another line", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot change"):
        seal_file(target, lock, today="2026-11-01")
    with pytest.raises(ValueError, match="is not a file"):
        seal_file(tmp_path / "missing.jsonl", lock)


def test_the_status_lists_sealed_files_and_whether_they_match(tmp_path, monkeypatch):
    lock = tmp_path / "final.lock.json"
    seal(lock, "gold_v9", "gold_v9.jsonl", "0" * 64, "2026-10-22")
    monkeypatch.setattr("adapt.eval.declare_claim.ROOT", tmp_path)
    assert sealed_lines(lock) == ["sealed 2026-10-22: gold_v9 000000000000 (missing)"]
    (tmp_path / "gold_v9.jsonl").write_text("x", encoding="utf-8")
    assert sealed_lines(lock)[0].endswith("(CHANGED)")


def test_the_committed_lock_seals_gold_v3_and_the_file_has_not_changed():
    from adapt.eval.runner import ROOT, hash_file

    sealed = read_sealed(RESULTS_ROOT / "final.lock.json")
    assert sealed["gold_v3"]["file"] == "platforms/tax-booking/evals/gold_v3.jsonl"
    assert len(sealed["gold_v3"]["sha256"]) == 64 and sealed["gold_v3"]["sealed"]
    path = ROOT / sealed["gold_v3"]["file"]
    if not path.is_file():
        pytest.skip("gold_v3.jsonl is not in this checkout yet")
    assert hash_file(path) == sealed["gold_v3"]["sha256"]
