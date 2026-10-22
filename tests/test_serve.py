"""The serve config and the launcher: the command that is built, the checks before it starts, and
nothing that needs vLLM or the network."""

import copy
import shutil
import subprocess
import sys

import pytest
import yaml

from adapt import serve
from adapt.eval.final_lock import read_claims
from adapt.eval.runner import ADAPTER_FILES, RESULTS_ROOT, ROOT, hash_directory
from adapt.models import Registry, load_registry
from adapt.serve import (
    SERVE_FILE,
    ServeConfig,
    check_adapters,
    environment,
    fetch_weights,
    load_serve_config,
    main,
    serve_command,
)
from adapt.train.config import load_train_config


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry()


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(SERVE_FILE.read_text(encoding="utf-8"))


@pytest.fixture
def adapter_folder(tmp_path):
    folder = tmp_path / "adapter"
    folder.mkdir()
    (folder / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    return folder


def config_with(raw, **changes) -> ServeConfig:
    return ServeConfig.model_validate({**copy.deepcopy(raw), **changes})


def test_the_committed_file_is_valid_and_names_a_model_of_the_registry(registry):
    config = load_serve_config()
    assert config.model in registry.models
    assert config.dtype == "half"
    assert {adapter.name for adapter in config.adapters} == {"tax-booking"}


def test_the_lora_rank_is_the_rank_the_adapter_was_trained_with():
    assert load_serve_config().lora.max_lora_rank == load_train_config().lora.r


def test_the_adapter_of_the_file_is_the_shipped_adapter_of_the_claim():
    claim = read_claims(RESULTS_ROOT / "final.lock.json")[0]
    hashes = {a["adapter_hash"] for c in claim["configurations"] for a in c["adapters"]}
    assert {adapter.sha256 for adapter in load_serve_config().adapters} <= hashes


def test_the_command_serves_the_pinned_revision_with_the_adapters_offline(registry):
    config = load_serve_config()
    command = serve_command(config, registry)
    entry = registry.models[config.model]
    assert command[:5] == [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", entry.hf_id]
    for flag, value in [
        ("--revision", entry.revision),
        ("--served-model-name", config.model),
        ("--dtype", "half"),
        ("--max-model-len", str(config.max_model_len)),
        ("--port", str(config.port)),
        ("--max-loras", "2"),
        ("--max-lora-rank", "16"),
    ]:
        assert command[command.index(flag) + 1] == value
    assert "--enable-lora" in command and "--enable-prefix-caching" in command
    modules = command[command.index("--lora-modules") + 1 :]
    assert len(modules) == 1 and modules[0].startswith("tax-booking=")


def test_prefix_caching_can_be_switched_off(raw, registry):
    command = serve_command(config_with(raw, prefix_caching=False), registry)
    assert "--enable-prefix-caching" not in command


def test_the_environment_turns_off_usage_stats_and_the_network():
    config = load_serve_config()
    env = environment(config, {"PATH": "/bin", "HF_HUB_OFFLINE": "0"})
    assert env["PATH"] == "/bin"
    assert env["VLLM_NO_USAGE_STATS"] == "1" and env["DO_NOT_TRACK"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1"


def test_an_unknown_model_is_refused(raw, registry):
    with pytest.raises(ValueError, match="unknown model 'gpt-9'.*qwen3-4b"):
        serve_command(config_with(raw, model="gpt-9"), registry)


def test_more_adapters_than_slots_are_refused(raw):
    second = {**raw["adapters"][0], "name": "second"}
    with pytest.raises(ValueError, match="2 adapters do not fit in 1 slots"):
        config_with(raw, adapters=[*raw["adapters"], second], lora={**raw["lora"], "max_loras": 1})


def test_the_same_adapter_name_twice_is_refused(raw):
    with pytest.raises(ValueError, match="used twice"):
        config_with(raw, adapters=[raw["adapters"][0], raw["adapters"][0]])


def test_an_unknown_key_in_the_file_is_refused(raw):
    with pytest.raises(ValueError):
        config_with(raw, port_number=8000)


def test_an_invalid_file_names_itself(tmp_path, raw):
    path = tmp_path / "serve.yaml"
    path.write_text(yaml.safe_dump({**raw, "port": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="serve.yaml is not valid"):
        load_serve_config(path)


def test_an_adapter_with_the_expected_hash_passes(raw, adapter_folder):
    expected = hash_directory(adapter_folder, ADAPTER_FILES)
    adapter = {"name": "tax-booking", "path": str(adapter_folder), "sha256": expected}
    check_adapters(config_with(raw, adapters=[adapter]))


def test_an_adapter_with_other_weights_is_refused(raw, adapter_folder):
    expected = hash_directory(adapter_folder, ADAPTER_FILES)
    adapter = {"name": "tax-booking", "path": str(adapter_folder), "sha256": expected}
    (adapter_folder / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="has hash .* the file expects"):
        check_adapters(config_with(raw, adapters=[adapter]))


def test_a_missing_adapter_is_refused(raw, tmp_path):
    adapter = {"name": "tax-booking", "path": str(tmp_path / "none"), "sha256": "0" * 64}
    with pytest.raises(ValueError, match="no adapter tax-booking in"):
        check_adapters(config_with(raw, adapters=[adapter]))


def test_the_weights_are_fetched_at_the_pinned_revision(registry, monkeypatch, tmp_path):
    calls = []

    def fake_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    assert fetch_weights(registry, "qwen3-4b") == tmp_path
    entry = registry.models["qwen3-4b"]
    assert calls == [
        (entry.hf_id, {"revision": entry.revision, "allow_patterns": list(serve.WEIGHT_FILES)})
    ]


def test_print_shows_the_command_and_starts_nothing(monkeypatch, capsys):
    def started(*args, **kwargs):
        raise AssertionError("vLLM must not start")

    monkeypatch.setattr(subprocess, "run", started)
    assert main(["--print"]) == 0
    output = capsys.readouterr().out
    assert "vllm.entrypoints.cli.main serve" in output and "HF_HUB_OFFLINE=1" in output


def test_the_launcher_stops_before_vllm_when_the_adapter_is_missing(raw, tmp_path, monkeypatch):
    raw["adapters"][0]["path"] = str(tmp_path / "none")
    path = tmp_path / "serve.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    def started(*args, **kwargs):
        raise AssertionError("vLLM must not start")

    monkeypatch.setattr(subprocess, "run", started)
    with pytest.raises(SystemExit, match="no adapter tax-booking"):
        main(["--file", str(path)])


def test_the_launcher_runs_the_command_with_the_environment(
    raw, adapter_folder, tmp_path, monkeypatch
):
    expected = hash_directory(adapter_folder, ADAPTER_FILES)
    raw["adapters"] = [{"name": "tax-booking", "path": str(adapter_folder), "sha256": expected}]
    path = tmp_path / "serve.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    seen = {}

    def fake_run(command, env, check):
        seen.update(command=command, env=env, check=check)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert main(["--file", str(path)]) == 0
    assert seen["command"][3] == "serve" and seen["env"]["HF_HUB_OFFLINE"] == "1"


needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


@needs_make
def test_the_make_targets_call_the_launcher():
    def planned(target, *variables):
        result = subprocess.run(
            ["make", "--dry-run", target, *variables],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    assert "-m adapt.serve --fetch-weights" in planned("fetch-weights")
    assert "-m adapt.serve" in planned("serve-local")
    assert "--print" in planned("serve-local", "PRINT=1")
    assert "--print" not in planned("serve-local")
