import hashlib
import json
import socket
import subprocess

import httpx
import pytest
import yaml

from adapt.baseline import (
    MAX_MODEL_LEN,
    fake_command,
    free_port,
    main,
    run_baseline,
    serve_command,
    system_name,
    wait_until_ready,
)
from adapt.config import load_platform
from adapt.eval.runner import SPLITS, results_file
from adapt.models import MODELS_FILE, Registry, load_registry
from adapt.prompting import system_prompt

RULES = load_platform("tax-booking").rules


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry()


class Process:
    """A stand-in for a server process. `code` is None while it is running."""

    def __init__(self, code=None):
        self.returncode = code
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


def test_the_server_command_pins_the_model_and_its_revision(registry):
    entry = registry.models["qwen3-4b"]
    command = serve_command(registry, "qwen3-4b", 8123)
    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3-4B"]
    pairs = dict(zip(command[3::2], command[4::2], strict=True))
    assert pairs == {
        "--revision": entry.revision,
        "--served-model-name": "qwen3-4b",
        "--dtype": "half",
        "--max-model-len": str(MAX_MODEL_LEN),
        "--gpu-memory-utilization": "0.9",
        "--seed": "0",
        "--port": "8123",
    }


def test_each_model_is_served_under_its_own_key(registry):
    for key, entry in registry.models.items():
        command = serve_command(registry, key, 8000)
        assert entry.hf_id in command and key in command and entry.revision in command


def test_the_context_length_holds_the_few_shot_prompt_and_the_longest_document():
    prompt = system_prompt("tax-booking", "SKR03", RULES, few_shot=True)
    # Three characters per token is a loose estimate. The answer needs about 400 tokens.
    assert (len(prompt) + 6000) / 3 + 400 < MAX_MODEL_LEN


def test_the_system_name_marks_the_few_shot_run():
    assert system_name("qwen3-4b") == "qwen3-4b-fewshot"


def test_a_free_port_can_be_used():
    port = free_port()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def test_waiting_ends_when_the_server_answers(server):
    server.version = "0.test"
    base = server.url.removesuffix("/v1")
    wait_until_ready(base, Process(), timeout=5.0, interval=0.01)


def test_waiting_gives_up_after_the_timeout(server):
    # No version is set, so the server answers 404 on /version.
    base = server.url.removesuffix("/v1")
    with pytest.raises(RuntimeError, match="not ready after 0 seconds"):
        wait_until_ready(base, Process(), timeout=0.2, interval=0.05)


def test_waiting_stops_at_once_when_the_process_has_ended():
    with pytest.raises(RuntimeError, match="exit code 3"):
        wait_until_ready("http://127.0.0.1:9", Process(code=3), timeout=60.0, interval=0.01)


def test_waiting_keeps_trying_while_nothing_listens_yet():
    port = free_port()
    with pytest.raises(RuntimeError, match="not ready"):
        wait_until_ready(f"http://127.0.0.1:{port}", Process(), timeout=0.3, interval=0.05)


def test_a_model_that_fails_a_gate_is_refused_before_any_server_starts(
    registry, tmp_path, monkeypatch
):
    with MODELS_FILE.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    data["models"]["qwen3-4b"]["licence"] = "gpl-3.0"
    changed = Registry.model_validate(data)

    def no_process(*args, **kwargs):
        raise AssertionError("a server was started")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    with pytest.raises(ValueError, match="qwen3-4b fails the model gates: licence"):
        run_baseline(changed, "qwen3-4b", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_server_with_another_vllm_version_is_refused_and_stopped(
    registry, server, tmp_path, monkeypatch
):
    server.version = "0.0.1"
    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: process)
    port = int(server.url.rsplit(":", 1)[1].split("/")[0])
    with pytest.raises(ValueError, match=r"runs vllm 0\.0\.1, the registry pins 0\.30\.0"):
        run_baseline(registry, "qwen3-4b", tmp_path, port=port)
    assert process.stopped
    assert list(tmp_path.iterdir()) == []


def test_the_server_is_stopped_when_it_never_becomes_ready(registry, tmp_path, monkeypatch):
    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: process)
    with pytest.raises(RuntimeError, match="not ready"):
        run_baseline(registry, "qwen3-4b", tmp_path, port=free_port(), timeout=0.2)
    assert process.stopped


def test_the_command_refuses_a_model_that_is_not_in_the_registry(tmp_path):
    with pytest.raises(SystemExit) as stop:
        main(["--model", "gpt-9", "--out-dir", str(tmp_path)])
    message = str(stop.value)
    assert "unknown model 'gpt-9'" in message and "qwen3-1.7b" in message


def test_the_fake_server_command_runs_the_script_of_the_tests():
    command = fake_command(8123)
    assert command[1].endswith("fake_serve.py") and command[2:] == ["--port", "8123"]


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory, registry):
    out_dir = tmp_path_factory.mktemp("eval")
    port = free_port()
    assert (
        main(["--model", "qwen3-1.7b", "--dry-run", "--port", str(port), "--out-dir", str(out_dir)])
        == 0
    )
    return out_dir, port


def test_a_dry_run_writes_the_results_of_every_split(dry_run):
    out_dir, _ = dry_run
    folder = out_dir / "tax-booking" / "qwen3-1.7b-fewshot"
    assert sorted(path.name for path in folder.iterdir()) == sorted(f"{s}.json" for s in SPLITS)
    for split in SPLITS:
        assert results_file(out_dir, "tax-booking", "qwen3-1.7b-fewshot", split).exists()


def test_a_dry_run_uses_the_few_shot_prompt_and_the_pinned_revision(dry_run, registry):
    out_dir, _ = dry_run
    prompt = system_prompt("tax-booking", "SKR03", RULES, few_shot=True)
    for split in SPLITS:
        path = results_file(out_dir, "tax-booking", "qwen3-1.7b-fewshot", split)
        manifest = json.loads(path.read_text(encoding="utf-8"))["manifest"]
        assert manifest["few_shot"] is True
        assert manifest["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert manifest["model"] == "qwen3-1.7b"
        assert manifest["model_revision"] == registry.models["qwen3-1.7b"].revision
        assert manifest["vllm_version"] == "fake", "a dry run is marked by its server version"


def test_a_dry_run_tunes_the_threshold_on_dev_and_reuses_it(dry_run):
    out_dir, _ = dry_run
    folder = out_dir / "tax-booking" / "qwen3-1.7b-fewshot"
    dev = json.loads((folder / "dev.json").read_text(encoding="utf-8"))
    assert dev["confidence_threshold"]["target_reached"] is True
    for split in SPLITS[1:]:
        results = json.loads((folder / f"{split}.json").read_text(encoding="utf-8"))
        assert results["manifest"]["threshold"] == dev["manifest"]["threshold"]


def test_a_dry_run_stops_its_server(dry_run):
    _, port = dry_run
    with pytest.raises(httpx.TransportError):
        httpx.get(f"http://127.0.0.1:{port}/version", timeout=2.0)


def test_the_gate_reads_the_results_of_a_dry_run(dry_run, capsys):
    from adapt.eval.gate import main as gate

    out_dir, _ = dry_run
    code = gate(["--system", "qwen3-1.7b-fewshot", "--out-dir", str(out_dir)])
    out = capsys.readouterr().out
    # Without a baseline, the comparison with few-shot is the one check that cannot pass.
    assert code == 1 and "gate: FAIL (1 of 10 checks)" in out
    assert "automation rate" in out and "not run: no baseline given" in out
