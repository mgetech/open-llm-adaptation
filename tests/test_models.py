import copy

import pytest
import yaml
from pydantic import ValidationError

from adapt import models
from adapt.models import (
    MODELS_FILE,
    HubInfo,
    Registry,
    check_hub,
    check_model,
    load_registry,
    pin_revisions,
    run_checks,
    training_memory_gb,
)

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry()


@pytest.fixture
def data() -> dict:
    with MODELS_FILE.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def results_by_name(entry, gates):
    return {r.name: r for r in check_model(entry, gates)}


def test_the_three_qwen3_models_are_listed(registry):
    assert {key: entry.hf_id for key, entry in registry.models.items()} == {
        "qwen3-1.7b": "Qwen/Qwen3-1.7B",
        "qwen3-4b": "Qwen/Qwen3-4B",
        "qwen3-8b": "Qwen/Qwen3-8B",
    }
    roles = {key: entry.role for key, entry in registry.models.items()}
    assert roles == {"qwen3-1.7b": "candidate", "qwen3-4b": "candidate", "qwen3-8b": "reference"}


def test_thinking_is_off_for_every_model(registry):
    assert all(not entry.chat_template.enable_thinking for entry in registry.models.values())


def test_every_model_passes_the_gates_in_the_file(registry):
    for key, entry in registry.models.items():
        failed = [r for r in check_model(entry, registry.gates) if not r.passed]
        assert not failed, (key, failed)


def test_every_revision_in_the_file_is_a_commit_sha(registry):
    for entry in registry.models.values():
        assert models.COMMIT_SHA.match(entry.revision), entry.hf_id


def test_a_disallowed_licence_fails_the_gate(registry, data):
    data["models"]["qwen3-4b"]["licence"] = "gpl-3.0"
    changed = Registry.model_validate(data)
    result = results_by_name(changed.models["qwen3-4b"], changed.gates)["licence"]
    assert not result.passed and "gpl-3.0" in result.detail
    assert results_by_name(changed.models["qwen3-1.7b"], changed.gates)["licence"].passed


def test_a_licence_that_only_looks_similar_fails(registry, data):
    for licence in ("apache-2.0-with-conditions", "apache", "unknown", ""):
        data["models"]["qwen3-4b"]["licence"] = licence
        changed = Registry.model_validate(data)
        result = results_by_name(changed.models["qwen3-4b"], changed.gates)["licence"]
        assert not result.passed, licence


def test_the_licence_is_compared_without_regard_to_case(registry, data):
    data["models"]["qwen3-4b"]["licence"] = "Apache-2.0"
    changed = Registry.model_validate(data)
    assert results_by_name(changed.models["qwen3-4b"], changed.gates)["licence"].passed


def test_the_allowlist_can_be_changed_in_the_file(data):
    data["gates"]["licences"] = ["mit"]
    changed = Registry.model_validate(data)
    result = results_by_name(changed.models["qwen3-4b"], changed.gates)["licence"]
    assert not result.passed


def test_the_memory_estimate_matches_the_hand_calculation(registry):
    gate = registry.gates.qlora
    # 4e9 parameters: 4e9 * 0.6 = 2.4 GB of weights,
    # 4e9 * 0.02 * 10 = 0.8 GB of trainable states, and 5.0 GB of activations at length 1,024.
    assert training_memory_gb(4_000_000_000, gate) == pytest.approx(8.2)


def test_a_longer_sequence_needs_more_memory(registry, data):
    short = training_memory_gb(4_000_000_000, registry.gates.qlora)
    data["gates"]["qlora"]["sequence_length"] = 2048
    long = training_memory_gb(4_000_000_000, Registry.model_validate(data).gates.qlora)
    assert long - short == pytest.approx(5.0)


def test_a_model_too_big_for_one_gpu_fails_the_size_gate(registry, data):
    data["models"]["qwen3-4b"]["parameters"] = 14_000_000_000
    changed = Registry.model_validate(data)
    result = results_by_name(changed.models["qwen3-4b"], changed.gates)["qlora fits one gpu"]
    assert not result.passed
    # 14e9 * 0.6 + 14e9 * 0.02 * 10 + 5e9 = 16.2 GB, above 16 GB * 0.9 = 14.4 GB.
    assert "16.2" in result.detail and "14.4" in result.detail


def test_the_size_gate_decides_at_the_limit(registry, data):
    # 0.6 + 0.2 GB per billion parameters, plus 5 GB, is exactly 14.4 GB at 11.75 billion.
    data["models"]["qwen3-4b"]["parameters"] = 11_750_000_000
    at_limit = Registry.model_validate(data)
    assert results_by_name(at_limit.models["qwen3-4b"], at_limit.gates)["qlora fits one gpu"].passed
    data["models"]["qwen3-4b"]["parameters"] = 11_750_001_000
    over = Registry.model_validate(data)
    assert not results_by_name(over.models["qwen3-4b"], over.gates)["qlora fits one gpu"].passed


def test_a_smaller_gpu_fails_a_model_that_fits_a_bigger_one(registry, data):
    data["gates"]["qlora"]["gpu_memory_gb"] = 8
    small = Registry.model_validate(data)
    assert not results_by_name(small.models["qwen3-4b"], small.gates)["qlora fits one gpu"].passed
    assert results_by_name(small.models["qwen3-1.7b"], small.gates)["qlora fits one gpu"].passed


def test_a_reference_model_is_not_held_to_the_training_gates(registry, data):
    data["models"]["qwen3-8b"]["parameters"] = 70_000_000_000
    data["models"]["qwen3-8b"]["architecture"] = "SomethingElse"
    changed = Registry.model_validate(data)
    names = results_by_name(changed.models["qwen3-8b"], changed.gates)
    assert set(names) == {"licence", "revision pinned"}
    assert all(r.passed for r in names.values())


def test_an_architecture_without_lora_support_fails(registry, data):
    data["models"]["qwen3-4b"]["architecture"] = "MambaForCausalLM"
    changed = Registry.model_validate(data)
    result = results_by_name(changed.models["qwen3-4b"], changed.gates)["vllm serves it with lora"]
    assert not result.passed and "MambaForCausalLM" in result.detail


def test_a_revision_that_is_not_a_full_sha_is_not_pinned(registry, data):
    for revision in ("main", "v1.0", SHA[:7], SHA.upper(), SHA + "0"):
        data["models"]["qwen3-4b"]["revision"] = revision
        changed = Registry.model_validate(data)
        result = results_by_name(changed.models["qwen3-4b"], changed.gates)["revision pinned"]
        assert not result.passed, revision


def test_unknown_fields_and_missing_fields_are_refused(data):
    extra = copy.deepcopy(data)
    extra["models"]["qwen3-4b"]["colour"] = "blue"
    with pytest.raises(ValidationError):
        Registry.model_validate(extra)
    missing = copy.deepcopy(data)
    del missing["models"]["qwen3-4b"]["licence"]
    with pytest.raises(ValidationError):
        Registry.model_validate(missing)
    with pytest.raises(ValidationError):
        Registry.model_validate({**data, "models": {}})


def test_the_role_must_be_known(data):
    data["models"]["qwen3-4b"]["role"] = "maybe"
    with pytest.raises(ValidationError):
        Registry.model_validate(data)


def entry_with(registry, **changes):
    return registry.models["qwen3-4b"].model_copy(update=changes)


def fake_hub(info):
    calls = []

    def hub(hf_id, revision):
        calls.append((hf_id, revision))
        return info

    hub.calls = calls
    return hub


def test_the_hub_check_passes_when_the_hub_agrees(registry):
    entry = registry.models["qwen3-4b"]
    hub = fake_hub(HubInfo(entry.revision, "apache-2.0", entry.parameters))
    results = check_hub(entry, hub)
    assert [r.name for r in results] == ["hub revision", "hub licence", "hub parameters"]
    assert all(r.passed for r in results)
    assert hub.calls == [("Qwen/Qwen3-4B", entry.revision)]


def test_each_difference_with_the_hub_fails_on_its_own(registry):
    entry = registry.models["qwen3-4b"]
    agree = HubInfo(entry.revision, "apache-2.0", entry.parameters)
    cases = {
        "hub revision": HubInfo(OTHER_SHA, "apache-2.0", entry.parameters),
        "hub licence": HubInfo(entry.revision, "other", entry.parameters),
        "hub parameters": HubInfo(entry.revision, "apache-2.0", entry.parameters + 1),
    }
    for name, info in cases.items():
        failed = [r.name for r in check_hub(entry, fake_hub(info)) if not r.passed]
        assert failed == [name]
    assert all(r.passed for r in check_hub(entry, fake_hub(agree)))


def test_a_hub_without_licence_or_size_fails(registry):
    entry = registry.models["qwen3-4b"]
    failed = [
        r.name
        for r in check_hub(entry, fake_hub(HubInfo(entry.revision, None, None)))
        if not r.passed
    ]
    assert failed == ["hub licence", "hub parameters"]


def test_a_hub_that_cannot_be_reached_fails_the_check(registry):
    def broken(hf_id, revision):
        raise ConnectionError("no route")

    results = check_hub(registry.models["qwen3-4b"], broken)
    assert len(results) == 1 and not results[0].passed
    assert "ConnectionError" in results[0].detail and "no route" in results[0].detail


def test_a_revision_that_is_not_a_sha_is_not_sent_to_the_hub(registry):
    hub = fake_hub(HubInfo(SHA, "apache-2.0", 1))
    assert check_hub(entry_with(registry, revision="main"), hub) == []
    assert hub.calls == []


def test_checks_for_all_models_can_skip_the_hub(registry):
    results = run_checks(registry, None)
    assert set(results) == set(registry.models)
    assert sum(len(found) for found in results.values()) == 10
    assert all(r.passed for found in results.values() for r in found)


def write_registry(tmp_path, data, text=None):
    path = tmp_path / "models.yaml"
    path.write_text(text if text is not None else yaml.safe_dump(data), encoding="utf-8")
    return path


def test_pinning_replaces_only_the_revisions_that_are_not_shas(tmp_path):
    text = MODELS_FILE.read_text(encoding="utf-8")
    text = text.replace("70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "main")
    text = text.replace("b968826d9c46dd6066d109eabc6255188de91218", "v1")
    path = write_registry(tmp_path, None, text)
    asked = []

    def hub(hf_id, revision):
        asked.append((hf_id, revision))
        return HubInfo(SHA if revision == "main" else OTHER_SHA, "apache-2.0", 1)

    assert pin_revisions(path, hub) == {"qwen3-1.7b": SHA, "qwen3-8b": OTHER_SHA}
    assert asked == [("Qwen/Qwen3-1.7B", "main"), ("Qwen/Qwen3-8B", "v1")]

    pinned = load_registry(path)
    assert pinned.models["qwen3-1.7b"].revision == SHA
    assert pinned.models["qwen3-8b"].revision == OTHER_SHA
    assert pinned.models["qwen3-4b"].revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    # Nothing else in the file changed, and the comments are still there.
    expected = text.replace("revision: main", f"revision: {SHA}")
    expected = expected.replace("revision: v1", f"revision: {OTHER_SHA}")
    assert path.read_text(encoding="utf-8") == expected
    assert "# Candidate and reference models" in expected


def test_pinning_does_nothing_when_all_revisions_are_shas(tmp_path):
    path = write_registry(tmp_path, None, MODELS_FILE.read_text(encoding="utf-8"))
    before = path.read_text(encoding="utf-8")
    hub = fake_hub(HubInfo(SHA, "apache-2.0", 1))
    assert pin_revisions(path, hub) == {}
    assert hub.calls == [] and path.read_text(encoding="utf-8") == before


def test_pinning_keeps_windows_line_endings(tmp_path):
    text = MODELS_FILE.read_text(encoding="utf-8").replace(
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "main"
    )
    path = tmp_path / "models.yaml"
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    pin_revisions(path, fake_hub(HubInfo(SHA, "apache-2.0", 1)))
    raw = path.read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n")
    assert f"revision: {SHA}\r\n".encode() in raw


def test_the_command_passes_for_the_file_in_the_repo(capsys):
    assert models.main(["--offline"]) == 0
    out = capsys.readouterr().out
    assert "models: PASS (10 of 10 checks)" in out
    for key in ("qwen3-1.7b", "qwen3-4b", "qwen3-8b"):
        assert key in out
    assert "FAIL" not in out


def test_the_command_asks_the_hub_unless_told_not_to(monkeypatch, capsys, registry):
    asked = []

    def hub(hf_id, revision):
        asked.append(hf_id)
        entry = next(e for e in registry.models.values() if e.hf_id == hf_id)
        return HubInfo(revision, entry.licence, entry.parameters)

    monkeypatch.setattr(models, "hub_info", hub)
    assert models.main([]) == 0
    assert sorted(asked) == ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
    assert "models: PASS (19 of 19 checks)" in capsys.readouterr().out
    asked.clear()
    assert models.main(["--offline"]) == 0
    assert asked == []


def test_the_command_fails_for_a_disallowed_licence(tmp_path, data, capsys):
    data["models"]["qwen3-4b"]["licence"] = "gpl-3.0"
    path = write_registry(tmp_path, data)
    assert models.main(["--offline", "--file", str(path)]) == 1
    out = capsys.readouterr().out
    assert "FAIL  licence: gpl-3.0" in out
    assert "models: FAIL (1 of 10 checks)" in out


def test_the_command_fails_when_the_hub_disagrees(monkeypatch, capsys):
    monkeypatch.setattr(models, "hub_info", lambda hf_id, revision: HubInfo(OTHER_SHA, "mit", 5))
    assert models.main([]) == 1
    out = capsys.readouterr().out
    assert "FAIL  hub revision" in out and "FAIL  hub licence" in out


def test_the_command_reports_a_file_it_cannot_read(tmp_path, data, capsys):
    del data["gates"]
    path = write_registry(tmp_path, data)
    assert models.main(["--offline", "--file", str(path)]) == 2
    assert "cannot read" in capsys.readouterr().err
    assert models.main(["--offline", "--file", str(tmp_path / "missing.yaml")]) == 2


def test_the_command_pins_and_then_checks(tmp_path, monkeypatch, capsys):
    text = MODELS_FILE.read_text(encoding="utf-8").replace(
        "1cfa9a7208912126459214e8b04321603b3df60c", "main"
    )
    path = write_registry(tmp_path, None, text)
    registry = load_registry(MODELS_FILE)

    def hub(hf_id, revision):
        entry = next(e for e in registry.models.values() if e.hf_id == hf_id)
        sha = entry.revision if revision == "main" else revision
        return HubInfo(sha, entry.licence, entry.parameters)

    monkeypatch.setattr(models, "hub_info", hub)
    assert models.main(["--pin", "--file", str(path)]) == 0
    out = capsys.readouterr().out
    assert "pinned qwen3-4b to 1cfa9a7208912126459214e8b04321603b3df60c" in out
    assert path.read_text(encoding="utf-8") == MODELS_FILE.read_text(encoding="utf-8")


def test_pinning_cannot_be_combined_with_offline(capsys):
    with pytest.raises(SystemExit) as stop:
        models.main(["--pin", "--offline"])
    assert stop.value.code == 2
