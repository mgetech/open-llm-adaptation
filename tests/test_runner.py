import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
from contextlib import ExitStack
from pathlib import Path

import pytest

from adapt.config import load_platform
from adapt.data.rounds import current_round
from adapt.eval.metrics import (
    Case,
    account_groups,
    accuracy_on_automated,
    automation_rate,
    critical_error_rate,
    document_accuracy,
    item_scores,
    review_recall,
    severity_counts,
    weighted_error_cost,
)
from adapt.eval.runner import (
    ADAPTER_FILES,
    DATA_DIR,
    ROOT,
    RunSettings,
    evaluate,
    git_state,
    hash_directory,
    load_canaries,
    load_documents,
    main,
    results_path,
    write_results,
)
from adapt.input_guard import injection_suspected
from adapt.pipeline import document_problems
from adapt.prompting import system_prompt
from fake_model import Model, unsure_about_4980, wrong_account_with_low_confidence
from fake_server import FakeServer, Reply

PLATFORM = load_platform("tax-booking")
GROUPS = account_groups(PLATFORM.rules)


def settings(server, split="dev", system="demo", **changes):
    values = dict(
        platform="tax-booking",
        system=system,
        split=split,
        endpoint=server.url,
        model="tax-booking",
        workers=4,
        threshold=0.0,
    )
    return RunSettings(**{**values, **changes})


def run(server, split="dev", **model_options):
    s = settings(server, split, **model_options.pop("settings", {}))
    server.handler = Model(
        load_documents("tax-booking", split), load_canaries("tax-booking"), **model_options
    )
    return evaluate(s)


@pytest.fixture(scope="module", autouse=True)
def no_input_guard():
    """These tests check the metrics of a perfect fake model on the real splits. The input guard
    would send their injection documents to review, so it is off here and has its own tests."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("adapt.pipeline.injection_suspected", lambda text: False)
        yield


@pytest.fixture(scope="module")
def perfect_dev():
    with FakeServer() as server:
        server.version = "0.test"
        yield run(server, "dev", settings={"threshold": None})


def test_a_perfect_system_scores_perfectly(perfect_dev):
    metrics = perfect_dev["metrics"]
    for name in ("document_accuracy", "automation_rate", "accuracy_on_automated", "review_recall"):
        assert metrics[name] == {"value": 1.0, "low": 1.0, "high": 1.0}, name
    for name in ("critical_error_rate", "weighted_error_cost", "invalid_json_rate"):
        assert metrics[name]["value"] == 0.0, name
    assert metrics["item_f1"]["value"] == 1.0
    assert metrics["severity_counts"] == {"critical": 0, "major": 0, "minor": 0}
    assert metrics["canary_leak_rate"] == {"value": 0.0, "documents": 5}


def test_the_injection_rate_is_defined_on_dev_because_it_has_injection_documents(perfect_dev):
    assert perfect_dev["metrics"]["injection_success_rate"]["value"] == 0.0


def test_the_manifest_records_what_was_run(perfect_dev):
    manifest = perfect_dev["manifest"]
    data_manifest = json.loads((ROOT / "results" / "data_manifest.json").read_text("utf-8"))
    prompt = system_prompt("tax-booking", "SKR03", PLATFORM.rules)
    assert manifest["model"] == "tax-booking"
    assert manifest["vllm_version"] == "0.test"
    assert manifest["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert manifest["data_sha256"] == data_manifest["splits"]["dev"]["sha256"]
    assert manifest["seed"] == 0
    assert manifest["threshold"] == 0.0
    assert manifest["documents"] == data_manifest["splits"]["dev"]["count"]
    assert manifest["model_revision"] is None and manifest["adapter_hash"] is None
    assert manifest["gpu"] is None or isinstance(manifest["gpu"], str)
    assert manifest["git_commit"] is None or re.fullmatch(r"[0-9a-f]{40}", manifest["git_commit"])
    assert manifest["git_dirty"] in (True, False, None)


def test_the_performance_block_has_the_measured_values(perfect_dev):
    performance = perfect_dev["performance"]
    assert performance["workers"] == 4
    assert performance["wall_seconds"] > 0
    assert performance["documents_per_second"] == pytest.approx(
        perfect_dev["manifest"]["documents"] / performance["wall_seconds"]
    )
    assert 0 < performance["latency_p50_seconds"] <= performance["latency_p95_seconds"]


def test_every_interval_contains_its_value(perfect_dev):
    for name, entry in perfect_dev["metrics"].items():
        if "low" in entry and entry["value"] is not None:
            assert entry["low"] <= entry["value"] <= entry["high"], name


def test_slices_cover_every_dimension_and_every_document(perfect_dev):
    slices = perfect_dev["slices"]
    assert set(slices) == {"items", "vat_rates", "layout", "noise", "supplier", "category"}
    n = perfect_dev["manifest"]["documents"]
    for dimension, values in slices.items():
        assert sum(v["n"] for v in values.values()) == n, dimension
    assert set(slices["vat_rates"]) <= {"none", "single", "mixed"}
    assert set(slices["items"]) <= {"0", "1", "2", "3", "4+"}
    assert set(slices["supplier"]) <= {"seen", "unseen"}
    for values in slices.values():
        for entry in values.values():
            assert set(entry) == {
                "n",
                "document_accuracy",
                "automation_rate",
                "accuracy_on_automated",
            }


def test_one_row_per_document_in_the_order_of_the_file(perfect_dev):
    ids = [d.id for d in load_documents("tax-booking", "dev")]
    assert [row["id"] for row in perfect_dev["documents"]] == ids
    row = perfect_dev["documents"][0]
    assert set(row) == {
        "id",
        "expected_review",
        "needs_review",
        "correct",
        "answer_correct",
        "severity",
        "reasons",
        "confidence",
        "item_confidences",
        "output",
        "by_rule",
    }
    assert all(row["correct"] for row in perfect_dev["documents"])


def test_every_row_keeps_the_model_output_and_the_item_confidences(perfect_dev):
    answered = [row for row in perfect_dev["documents"] if row["output"] is not None]
    assert answered
    for row in answered:
        assert set(row["output"]) >= {"items", "currency", "reverse_charge", "needs_review"}
        if row["output"]["items"]:
            assert len(row["item_confidences"]) == len(row["output"]["items"])
            assert row["confidence"] == min(row["item_confidences"])
        else:
            assert row["item_confidences"] is None


def test_the_results_file_is_valid_json_at_the_documented_path(tmp_path, perfect_dev):
    s = RunSettings("tax-booking", "my-system", "dev", "http://unused/v1", "m")
    path = results_path(s, tmp_path)
    assert path == tmp_path / "tax-booking" / "my-system" / "dev.json"
    write_results(perfect_dev, path)
    assert json.loads(path.read_text(encoding="utf-8")) == perfect_dev
    assert path.read_bytes().endswith(b"}\n")
    assert b"\r" not in path.read_bytes()


def test_the_numbers_match_the_metric_functions_for_a_flawed_system(server):
    documents = load_documents("tax-booking", "edge")

    def flip_vat(document):
        output = copy.deepcopy(dict(document.target))
        if int(document.id.split("-")[1]) % 3 == 0:
            for item in output["items"][:1]:
                item["vat_rate"] = "7" if item["vat_rate"] == "19" else "19"
        return output

    results = run(server, "edge", answer=flip_vat)
    cases = []
    for document, row in zip(documents, results["documents"], strict=True):
        output = flip_vat(document)
        cases.append(
            Case(
                target=document.target,
                expected_review=document.expected_review,
                needs_review=row["needs_review"],
                items=output["items"],
                reverse_charge=output["reverse_charge"],
                category=document.category,
            )
        )
    metrics = results["metrics"]
    assert metrics["document_accuracy"]["value"] == pytest.approx(document_accuracy(cases))
    assert metrics["automation_rate"]["value"] == pytest.approx(automation_rate(cases))
    assert metrics["accuracy_on_automated"]["value"] == pytest.approx(accuracy_on_automated(cases))
    assert metrics["critical_error_rate"]["value"] == pytest.approx(
        critical_error_rate(cases, GROUPS)
    )
    assert metrics["weighted_error_cost"]["value"] == pytest.approx(
        weighted_error_cost(cases, GROUPS)
    )
    assert metrics["review_recall"]["value"] == pytest.approx(review_recall(cases))
    assert metrics["item_f1"]["value"] == pytest.approx(item_scores(cases).f1)
    assert metrics["severity_counts"] == severity_counts(cases, GROUPS)
    assert metrics["critical_error_rate"]["value"] > 0
    for name, entry in metrics.items():
        if "low" in entry and entry["value"] is not None:
            assert entry["low"] <= entry["value"] <= entry["high"], name


def test_a_model_that_leaks_a_canary_name_is_caught(server):
    leaking = run(server, "edge", leak=True)
    assert leaking["metrics"]["canary_leak_rate"] == {"value": 1.0, "documents": 5}
    clean = run(server, "edge")
    assert clean["metrics"]["canary_leak_rate"]["value"] == 0.0


def test_a_model_that_is_down_sends_everything_to_review(server):
    results = run(server, "edge", answer=lambda document: Reply(status=500, body={}))
    assert results["metrics"]["automation_rate"]["value"] == 0.0
    assert results["metrics"]["review_recall"]["value"] == 1.0
    assert all(row["needs_review"] for row in results["documents"])
    reasons = {reason for row in results["documents"] for reason in row["reasons"]}
    assert "model_unavailable" in reasons


def test_requests_wait_longer_than_the_gateway_does(server, monkeypatch):
    """A busy server on a small GPU answers late. A timeout must not count as a model error."""
    from adapt.client import TIMEOUT_SECONDS, ModelClient
    from adapt.eval import runner

    seen = []

    class Recording(ModelClient):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("timeout"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(runner, "ModelClient", Recording)
    run(server, "gold")
    assert seen == [runner.REQUEST_TIMEOUT_SECONDS]
    assert runner.REQUEST_TIMEOUT_SECONDS > 2 * TIMEOUT_SECONDS


def test_answers_that_are_not_json_count_as_invalid(server):
    results = run(server, "edge", answer=lambda document: "not json")
    rows = results["documents"]
    invalid = sum("invalid_output" in row["reasons"] for row in rows)
    assert invalid > 0
    assert results["metrics"]["invalid_json_rate"]["value"] == pytest.approx(invalid / len(rows))
    assert results["metrics"]["automation_rate"]["value"] == 0.0


def test_a_low_confidence_sends_documents_to_review_at_the_given_threshold(server):
    def unsure(token):
        return -1.0 if token.isdigit() and len(token) == 4 else -0.001

    low = run(server, "edge", logprob_for=unsure, settings={"threshold": 0.5})
    assert low["manifest"]["threshold"] == 0.5
    automatable = [r for r in low["documents"] if not r["expected_review"]]
    assert any("low_confidence" in r["reasons"] for r in automatable)
    assert low["metrics"]["automation_rate"]["value"] < 1.0
    assert all(
        r["confidence"] is None or r["confidence"] == pytest.approx(math.exp(-1.0))
        for r in low["documents"]
    )
    high = run(server, "edge", logprob_for=unsure, settings={"threshold": 0.0})
    assert high["metrics"]["automation_rate"]["value"] == 1.0


def test_the_same_run_twice_gives_the_same_numbers(server):
    first = run(server, "edge")
    second = run(server, "edge")
    for key in ("metrics", "slices", "documents"):
        assert first[key] == second[key]


def test_the_gold_split_runs_without_noise_and_category_data(server):
    results = run(server, "gold")
    assert results["manifest"]["documents"] == 80
    assert len(results["documents"]) == 80
    assert "noise" not in results["slices"]
    assert set(results["slices"]["supplier"]) == {"unseen"}
    assert set(results["slices"]["layout"]) == {
        "table_invoice",
        "letter_invoice",
        "till_receipt",
        "short_receipt",
        "credit_note",
        "eu_invoice",
    }
    assert results["metrics"]["document_accuracy"]["value"] == 1.0
    assert results["metrics"]["injection_success_rate"]["value"] is None
    assert (
        results["manifest"]["data_sha256"]
        == hashlib.sha256(
            (ROOT / "platforms/tax-booking/evals/gold.jsonl").read_bytes()
        ).hexdigest()
    )


def test_the_command_line_writes_the_results_and_prints_a_summary(server, tmp_path, capsys):
    server.handler = Model(load_documents("tax-booking", "edge"), load_canaries("tax-booking"))
    code = main(
        [
            "--system",
            "cli-test",
            "--split",
            "edge",
            "--endpoints",
            server.url,
            "--workers",
            "2",
            "--threshold",
            "0",
            "--out-dir",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    path = tmp_path / "tax-booking" / "cli-test" / "edge.json"
    assert code == 0
    assert "cli-test on edge: 150 documents" in out
    assert "document_accuracy 1.000" in out
    assert "confidence threshold 0.0000 (given)" in out
    assert f"results: {path}" in out
    assert json.loads(path.read_text(encoding="utf-8"))["manifest"]["model"] == "cli-test"


def test_the_model_name_can_differ_from_the_system_name(server, tmp_path):
    server.handler = Model(load_documents("tax-booking", "edge"), load_canaries("tax-booking"))
    main(
        [
            "--system",
            "label",
            "--model",
            "adapter-a",
            "--split",
            "edge",
            "--endpoints",
            server.url,
            "--threshold",
            "0",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert {request.body["model"] for request in server.requests} == {"adapter-a"}


def test_a_system_name_must_be_a_plain_folder_name(server):
    for name in ("../escape", "a/b", "", ".hidden", "with space"):
        with pytest.raises(ValueError, match="not allowed"):
            evaluate(settings(server, system=name))


def test_an_unknown_split_is_refused():
    with pytest.raises(ValueError, match="unknown split"):
        load_documents("tax-booking", "train")


def test_the_system_name_is_required():
    with pytest.raises(SystemExit):
        main(["--split", "dev"])


def test_the_canary_documents_are_the_five_train_documents_without_their_names():
    canaries = load_canaries("tax-booking")
    assert len(canaries) == 5
    for canary in canaries:
        assert canary.name.split()[0].lower() not in canary.text.lower()


def test_loaded_documents_match_the_committed_split_files():
    for split, count in (("dev", 300), ("test_iid", 500), ("test_ood", 300), ("edge", 150)):
        documents = load_documents("tax-booking", split)
        assert len(documents) == count
        assert (DATA_DIR / "tax-booking" / f"round{current_round()}" / f"{split}.jsonl").exists()


def test_a_hash_of_a_folder_changes_with_its_files(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"one")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"two")
    first = hash_directory(tmp_path)
    assert hash_directory(tmp_path) == first
    (tmp_path / "sub" / "b.bin").write_bytes(b"changed")
    changed = hash_directory(tmp_path)
    assert changed != first
    (tmp_path / "sub" / "b.bin").write_bytes(b"two")
    (tmp_path / "sub" / "b.bin").rename(tmp_path / "sub" / "c.bin")
    assert hash_directory(tmp_path) not in (first, changed)
    with pytest.raises(ValueError, match="not a directory"):
        hash_directory(tmp_path / "missing")


def test_the_hash_of_a_folder_does_not_depend_on_the_system(tmp_path: Path):
    """Files are taken in the order of their names as text, so capital letters come first."""
    (tmp_path / "adapter_config.json").write_bytes(b"config")
    (tmp_path / "README.md").write_bytes(b"readme")
    digest = hashlib.sha256()
    for name, content in (("README.md", b"readme"), ("adapter_config.json", b"config")):
        digest.update(name.encode() + b"\0")
        digest.update(hashlib.sha256(content).hexdigest().encode() + b"\n")
    assert hash_directory(tmp_path) == digest.hexdigest()


def test_an_adapter_folder_is_hashed_into_the_manifest_by_its_two_files(server, tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "adapter_config.json").write_bytes(b"config")
    results = run(server, "edge", settings={"adapter_path": tmp_path, "model_revision": "abc123"})
    expected = results["manifest"]["adapter_hash"]
    assert expected == hash_directory(tmp_path, ADAPTER_FILES)
    assert results["manifest"]["model_revision"] == "abc123"
    # the download tool leaves its own notes next to the files; they are not the adapter
    (tmp_path / ".cache" / "huggingface").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "adapter_config.json.metadata").write_bytes(b"x")
    results = run(server, "edge", settings={"adapter_path": tmp_path, "model_revision": "abc123"})
    assert results["manifest"]["adapter_hash"] == expected


def test_on_dev_the_threshold_is_tuned_and_stored_with_a_coverage_table(server):
    results = run(
        server,
        "dev",
        answer=wrong_account_with_low_confidence(),
        logprob_for=unsure_about_4980,
        settings={"threshold": None},
    )
    tuning = results["confidence_threshold"]
    assert tuning["source"] == "tuned on dev"
    assert tuning["target_accuracy"] == PLATFORM.config.gate.accuracy_automated_min
    assert tuning["target_reached"] is True
    assert tuning["value"] == results["manifest"]["threshold"]
    # The wrong answers have a confidence of exp(-1.5), so the threshold must lie above it.
    assert tuning["value"] > math.exp(-1.5)
    assert tuning["accuracy_on_automated"] >= tuning["target_accuracy"]
    assert results["metrics"]["accuracy_on_automated"]["value"] >= tuning["target_accuracy"]
    assert results["metrics"]["accuracy_on_automated"]["value"] == pytest.approx(
        tuning["accuracy_on_automated"]
    )
    assert results["metrics"]["automation_rate"]["value"] == pytest.approx(
        tuning["automation_rate"]
    )
    assert 0.5 < tuning["automation_rate"] < 1.0
    curve = tuning["curve"]
    assert curve[0]["threshold"] == 0.0
    assert [p["threshold"] for p in curve] == sorted({p["threshold"] for p in curve})
    assert curve[0]["accuracy_on_automated"] < tuning["target_accuracy"]
    rows = results["documents"]
    wrong = [r for r in rows if not r["answer_correct"] and not r["expected_review"]]
    assert wrong and all(r["needs_review"] and "low_confidence" in r["reasons"] for r in wrong)


def test_a_threshold_that_is_given_is_not_tuned(server):
    results = run(server, "dev", settings={"threshold": 0.4})
    assert results["confidence_threshold"] == {"value": 0.4, "source": "given"}
    assert results["manifest"]["threshold"] == 0.4


def test_other_splits_use_the_threshold_tuned_on_dev_for_the_same_system(server, tmp_path):
    kwargs = dict(answer=wrong_account_with_low_confidence(), logprob_for=unsure_about_4980)
    dev = run(server, "dev", settings={"threshold": None, "out_dir": tmp_path}, **kwargs)
    write_results(dev, tmp_path / "tax-booking" / "demo" / "dev.json")
    edge = run(server, "edge", settings={"threshold": None, "out_dir": tmp_path}, **kwargs)
    assert edge["manifest"]["threshold"] == dev["confidence_threshold"]["value"]
    assert edge["confidence_threshold"]["value"] == dev["confidence_threshold"]["value"]
    assert edge["confidence_threshold"]["source"] == "tuned on dev (dev.json)"
    assert "curve" not in edge["confidence_threshold"]


def test_a_split_without_dev_results_asks_for_them(server, tmp_path, capsys):
    with pytest.raises(ValueError, match="run the dev split first"):
        run(server, "edge", settings={"threshold": None, "out_dir": tmp_path})
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "--system",
                "none-yet",
                "--split",
                "edge",
                "--endpoints",
                server.url,
                "--out-dir",
                str(tmp_path),
            ]
        )
    assert "run the dev split first" in str(stopped.value)


def test_when_dev_cannot_reach_the_target_there_is_no_confidence_limit(server, tmp_path):
    def always_wrong(document):
        output = copy.deepcopy(dict(document.target))
        for item in output["items"]:
            item["account"] = "4980" if item["account"] != "4980" else "4930"
        return output

    dev = run(server, "dev", answer=always_wrong, settings={"threshold": None, "out_dir": tmp_path})
    tuning = dev["confidence_threshold"]
    assert tuning["target_reached"] is False
    assert tuning["value"] is None and tuning["automation_rate"] is None
    assert tuning["source"] == "dev target not reached, no confidence limit"
    assert dev["manifest"]["threshold"] == 0.0
    assert len(tuning["curve"]) > 1
    write_results(dev, tmp_path / "tax-booking" / "demo" / "dev.json")
    edge = run(server, "edge", settings={"threshold": None, "out_dir": tmp_path})
    assert edge["manifest"]["threshold"] == 0.0
    assert edge["confidence_threshold"]["source"] == "dev target not reached, no confidence limit"


def serve_edge(count, answer=None, delays=None):
    """`count` fake servers that answer the edge split the same way. Returns them in a stack."""
    stack = ExitStack()
    servers = []
    for index in range(count):
        server = stack.enter_context(FakeServer())
        server.version = "0.test"
        model = Model(
            load_documents("tax-booking", "edge"), load_canaries("tax-booking"), answer=answer
        )
        delay = (delays or {}).get(index, 0.0)

        def handler(body, model=model, delay=delay):
            reply = model(body)
            reply.delay = delay
            return reply

        server.handler = handler
        servers.append(server)
    return stack, servers


def evaluate_edge(servers):
    endpoints = tuple(server.url for server in servers)
    return evaluate(settings(servers[0], "edge", endpoint=endpoints, workers=3))


@pytest.mark.parametrize("count", [2, 3, 4])
def test_several_servers_give_the_results_of_one_server(count):
    answer = wrong_account_with_low_confidence()
    stack, servers = serve_edge(count, answer)
    with stack:
        merged = evaluate_edge(servers)
        single = evaluate_edge(servers[:1])
    assert merged["documents"] == single["documents"]
    assert merged["metrics"] == single["metrics"]
    assert merged["slices"] == single["slices"]
    assert {**merged["manifest"], "servers": 1} == single["manifest"]
    assert merged["manifest"]["servers"] == count and single["manifest"]["servers"] == 1


@pytest.mark.parametrize("count", [2, 3, 4])
def test_the_documents_are_shared_round_robin_by_index(count):
    """Some edge documents are refused before they reach a model, so the requests of one server
    are compared with its share, and all of them with the requests of a single server."""
    documents = load_documents("tax-booking", "edge")
    canaries = {c.text for c in load_canaries("tax-booking")}
    stack, servers = serve_edge(count + 1)
    with stack:
        evaluate_edge(servers[:count])
        evaluate_edge(servers[count:])

    def asked(server):
        return {r.body["messages"][1]["content"] for r in server.requests} - canaries

    for index, server in enumerate(servers[:count]):
        assert asked(server) <= {d.text for d in documents[index::count]}
    assert set().union(*(asked(server) for server in servers[:count])) == asked(servers[count])
    assert all(asked(server) for server in servers[:count])


def test_the_results_stay_in_document_order_when_one_server_is_slower():
    documents = load_documents("tax-booking", "edge")
    stack, servers = serve_edge(2, delays={0: 0.03})
    with stack:
        results = evaluate_edge(servers)
    assert [row["id"] for row in results["documents"]] == [d.id for d in documents]


def test_each_server_has_its_own_requests_in_flight():
    stack, servers = serve_edge(2, delays={0: 0.02, 1: 0.02})
    with stack:
        results = evaluate_edge(servers)
    assert results["performance"]["workers"] == 3
    # The documents that fail a document check never reach a server.
    platform = load_platform("tax-booking")
    sent = sum(
        not document_problems(d.text, d.chart, platform)
        for d in load_documents("tax-booking", "edge")
    )
    assert all(len(server.requests) > 0.4 * sent for server in servers)


def test_servers_with_different_vllm_versions_are_refused_before_any_request():
    stack, servers = serve_edge(2)
    servers[1].version = "0.other"
    with stack, pytest.raises(ValueError, match="do not run the same vllm version"):
        evaluate_edge(servers)
    assert all(server.requests == [] for server in servers)


def test_a_run_needs_at_least_one_server():
    with pytest.raises(ValueError, match="at least one endpoint"):
        RunSettings("tax-booking", "demo", "dev", (), "m")


def test_one_endpoint_may_be_given_as_text_or_as_a_tuple():
    one = RunSettings("tax-booking", "demo", "dev", "http://a/v1", "m")
    many = RunSettings("tax-booking", "demo", "dev", ("http://a/v1", "http://b/v1"), "m")
    assert one.endpoints == ("http://a/v1",)
    assert many.endpoints == ("http://a/v1", "http://b/v1")


def test_the_command_line_takes_several_endpoints(tmp_path, capsys):
    stack, servers = serve_edge(2)
    with stack:
        code = main(
            ["--system", "two", "--split", "edge", "--endpoints", servers[0].url, servers[1].url]
            + ["--threshold", "0", "--out-dir", str(tmp_path)]
        )
    assert code == 0
    saved = json.loads((tmp_path / "tax-booking" / "two" / "edge.json").read_text(encoding="utf-8"))
    assert saved["manifest"]["servers"] == 2
    assert len(servers[0].requests) > 0 and len(servers[1].requests) > 0


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(folder, *arguments):
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run(["git", *identity, *arguments], cwd=folder, check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init")
    (tmp_path / "code.py").write_text("print(1)\n", encoding="utf-8")
    git(tmp_path, "add", "code.py")
    git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-m", "first")
    return tmp_path


@needs_git
def test_a_clean_tree_is_not_dirty(repository):
    commit, dirty = git_state(repository)
    assert len(commit) == 40 and dirty is False


@needs_git
def test_a_new_file_that_git_does_not_track_is_not_a_change(repository):
    """The results of an earlier run must not mark the next run as dirty."""
    (repository / "results.json").write_text("{}", encoding="utf-8")
    assert git_state(repository)[1] is False


@needs_git
def test_a_changed_file_that_git_tracks_is_a_change(repository):
    (repository / "code.py").write_text("print(2)\n", encoding="utf-8")
    assert git_state(repository)[1] is True


def test_no_git_gives_no_state(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert git_state(tmp_path) == (None, None)


def test_the_input_guard_stops_injection_documents_before_the_model(server, monkeypatch):
    monkeypatch.setattr("adapt.pipeline.injection_suspected", injection_suspected)
    results = run(server, "dev", settings={"threshold": 0.0})
    flagged = [r for r in results["documents"] if "injection_suspected" in r["reasons"]]
    assert flagged
    assert all(r["by_rule"] and r["needs_review"] and r["output"] is None for r in flagged)
    assert len(server.requests) <= len(results["documents"]) - len(flagged)
    assert results["metrics"]["injection_success_rate"]["value"] == 0.0
    assert results["metrics"]["injection_unexpected_rate"]["value"] > 0.0
