"""Evaluating a trained adapter. These tests use the fake server, so they need no GPU and no
GPU library. A folder with an `adapter_config.json` stands in for the adapter."""

import hashlib
import json
import subprocess

import httpx
import pytest
import yaml

from adapt.baseline import free_port, serve_command
from adapt.config import load_platform
from adapt.eval.gate import main as gate
from adapt.eval.runner import SPLITS, hash_directory, results_file
from adapt.models import MODELS_FILE, Registry, load_registry
from adapt.prompting import system_prompt
from adapt.train.evaluate import evaluate_adapter, lora_command, main, system_name

RULES = load_platform("tax-booking").rules


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry()


class Process:
    """A stand-in for a server process. `code` is None while it is running."""

    def __init__(self):
        self.returncode = None
        self.stopped = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.stopped = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.stopped = True


def make_adapter(adapters_dir, name):
    folder = adapters_dir / "tax-booking" / name
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    return folder


def test_the_system_name_has_the_model_and_the_seed():
    assert system_name("qwen3-4b", 2) == "qwen3-4b-sft-2"


def test_a_short_check_and_a_smoke_run_have_their_own_system_names():
    assert system_name("qwen3-4b", 0, max_steps=3) == "qwen3-4b-sft-0-steps3"
    assert system_name(None, 1, smoke=True) == "smoke-sft-1"


def test_the_server_command_is_the_baseline_command_with_the_adapter(registry, tmp_path):
    adapter = tmp_path / "qwen3-4b-0"
    command = lora_command(registry, "qwen3-4b", 8123, "qwen3-4b-sft-0", adapter, 16)
    base = serve_command(registry, "qwen3-4b", 8123)
    assert command[: len(base)] == base
    assert command[len(base) :] == [
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--lora-modules",
        f"qwen3-4b-sft-0={adapter.resolve()}",
    ]


def test_a_smoke_adapter_is_refused_without_a_dry_run(registry, tmp_path):
    make_adapter(tmp_path, "smoke-0")
    with pytest.raises(ValueError, match="random, evaluate it with --dry-run"):
        evaluate_adapter(registry, None, 0, tmp_path / "eval", adapters_dir=tmp_path, smoke=True)


def test_an_unknown_split_is_refused(registry, tmp_path):
    with pytest.raises(ValueError, match="unknown split 'train'"):
        evaluate_adapter(
            registry, "qwen3-4b", 0, tmp_path, adapters_dir=tmp_path, splits=["dev", "train"]
        )


def test_a_missing_adapter_is_refused_before_any_server_starts(registry, tmp_path, monkeypatch):
    def no_process(*args, **kwargs):
        raise AssertionError("a server was started")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    with pytest.raises(ValueError, match="no adapter in .*qwen3-4b-0, train it first"):
        evaluate_adapter(registry, "qwen3-4b", 0, tmp_path / "eval", adapters_dir=tmp_path)
    assert not (tmp_path / "eval").exists()


def test_a_model_that_fails_a_gate_is_refused_before_any_server_starts(tmp_path, monkeypatch):
    data = yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))
    data["models"]["qwen3-4b"]["licence"] = "gpl-3.0"
    changed = Registry.model_validate(data)
    make_adapter(tmp_path, "qwen3-4b-0")

    def no_process(*args, **kwargs):
        raise AssertionError("a server was started")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    with pytest.raises(ValueError, match="qwen3-4b fails the model gates: licence"):
        evaluate_adapter(changed, "qwen3-4b", 0, tmp_path / "eval", adapters_dir=tmp_path)


def test_a_server_with_another_vllm_version_is_refused_and_stopped(
    registry, server, tmp_path, monkeypatch
):
    server.version = "0.0.1"
    make_adapter(tmp_path, "qwen3-4b-0")
    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: process)
    port = int(server.url.rsplit(":", 1)[1].split("/")[0])
    with pytest.raises(ValueError, match=r"runs vllm 0\.0\.1, the registry pins 0\.30\.0"):
        evaluate_adapter(
            registry, "qwen3-4b", 0, tmp_path / "eval", adapters_dir=tmp_path, port=port
        )
    assert process.stopped
    assert not (tmp_path / "eval").exists()


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory, registry):
    root = tmp_path_factory.mktemp("sft")
    adapter = make_adapter(root, "qwen3-1.7b-0")
    port = free_port()
    arguments = ["--model", "qwen3-1.7b", "--dry-run", "--port", str(port)]
    arguments += ["--adapters-dir", str(root), "--out-dir", str(root / "eval")]
    assert main(arguments) == 0
    return root, adapter, port


def test_a_dry_run_writes_the_results_of_every_split(dry_run):
    root, _, _ = dry_run
    folder = root / "eval" / "tax-booking" / "qwen3-1.7b-sft-0"
    assert sorted(path.name for path in folder.iterdir()) == sorted(f"{s}.json" for s in SPLITS)


def test_the_manifest_records_the_adapter_and_the_zero_shot_prompt(dry_run, registry):
    root, adapter, _ = dry_run
    prompt = system_prompt("tax-booking", "SKR03", RULES)
    for split in SPLITS:
        path = results_file(root / "eval", "tax-booking", "qwen3-1.7b-sft-0", split)
        manifest = json.loads(path.read_text(encoding="utf-8"))["manifest"]
        assert manifest["few_shot"] is False
        assert manifest["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert manifest["adapter_hash"] == hash_directory(adapter)
        assert manifest["model_revision"] == registry.models["qwen3-1.7b"].revision
        assert manifest["vllm_version"] == "fake", "a dry run is marked by its server version"


def test_a_dry_run_tunes_the_threshold_on_dev_and_reuses_it(dry_run):
    root, _, _ = dry_run
    folder = root / "eval" / "tax-booking" / "qwen3-1.7b-sft-0"
    dev = json.loads((folder / "dev.json").read_text(encoding="utf-8"))
    for split in SPLITS[1:]:
        results = json.loads((folder / f"{split}.json").read_text(encoding="utf-8"))
        assert results["manifest"]["threshold"] == dev["manifest"]["threshold"]


def test_a_dry_run_stops_its_server(dry_run):
    _, _, port = dry_run
    with pytest.raises(httpx.TransportError):
        httpx.get(f"http://127.0.0.1:{port}/version", timeout=2.0)


def test_the_gate_reads_the_results_of_a_dry_run(dry_run, capsys):
    root, _, _ = dry_run
    code = gate(["--system", "qwen3-1.7b-sft-0", "--out-dir", str(root / "eval")])
    out = capsys.readouterr().out
    assert code == 1 and "gate: FAIL (1 of 10 checks)" in out
    assert "not run: no baseline given" in out


def test_a_smoke_adapter_is_evaluated_against_the_fake_server(tmp_path):
    make_adapter(tmp_path, "smoke-3")
    arguments = ["--smoke", "--seed", "3", "--dry-run", "--splits", "dev"]
    arguments += ["--adapters-dir", str(tmp_path), "--out-dir", str(tmp_path / "eval")]
    assert main(arguments) == 0
    path = results_file(tmp_path / "eval", "tax-booking", "smoke-sft-3", "dev")
    manifest = json.loads(path.read_text(encoding="utf-8"))["manifest"]
    assert manifest["model_revision"] is None and manifest["vllm_version"] == "fake"
    assert [p.name for p in path.parent.iterdir()] == ["dev.json"]


def test_a_short_check_uses_the_adapter_and_the_name_of_the_short_run(tmp_path):
    make_adapter(tmp_path, "qwen3-1.7b-0-steps3-gpus1")
    arguments = ["--model", "qwen3-1.7b", "--max-steps", "3", "--dry-run", "--splits", "dev"]
    arguments += ["--adapters-dir", str(tmp_path), "--out-dir", str(tmp_path / "eval")]
    assert main(arguments) == 0
    assert (tmp_path / "eval" / "tax-booking" / "qwen3-1.7b-sft-0-steps3" / "dev.json").exists()


def test_a_short_check_on_two_gpus_finds_its_adapter(tmp_path):
    make_adapter(tmp_path, "qwen3-1.7b-0-steps3-gpus2")
    arguments = ["--model", "qwen3-1.7b", "--max-steps", "3", "--num-gpus", "2", "--dry-run"]
    arguments += ["--splits", "dev", "--adapters-dir", str(tmp_path)]
    arguments += ["--out-dir", str(tmp_path / "eval")]
    assert main(arguments) == 0
    assert (tmp_path / "eval" / "tax-booking" / "qwen3-1.7b-sft-0-steps3" / "dev.json").exists()


def test_the_command_needs_a_model_unless_it_is_a_smoke_run():
    with pytest.raises(SystemExit) as stop:
        main([])
    assert stop.value.code == 2


def test_the_command_reports_a_missing_adapter(tmp_path):
    with pytest.raises(SystemExit) as stop:
        main(["--model", "qwen3-4b", "--adapters-dir", str(tmp_path), "--dry-run"])
    assert "train it first" in str(stop.value)
