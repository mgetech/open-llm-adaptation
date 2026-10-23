"""The compliance note names controls and files. They must exist and say what the note says."""

import json
import re

import yaml

from adapt.data.pseudonymise import PLACEHOLDERS
from adapt.eval.runner import ROOT

NOTE = (ROOT / "docs" / "compliance.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_every_file_the_note_names_exists():
    names = re.findall(r"`((?:docs|deploy|src|configs|results|tests)/[\w./-]+)`", NOTE)
    assert len(names) >= 15
    for name in names:
        assert (ROOT / name).exists(), name


def test_vllm_is_on_a_network_without_a_route_out_and_only_the_gateway_has_a_port():
    compose = yaml.safe_load((ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8"))
    internal = [name for name, network in compose["networks"].items() if network.get("internal")]
    assert internal and "internal: true" in NOTE
    assert "ports" not in compose["services"]["vllm"]
    assert compose["services"]["gateway"]["ports"]
    assert set(compose["services"]["vllm"]["networks"]) <= set(internal)


def test_serving_is_offline_and_without_telemetry():
    environment = yaml.safe_load((ROOT / "configs" / "serve.yaml").read_text("utf-8"))[
        "environment"
    ]
    for name in ("HF_HUB_OFFLINE", "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):
        assert environment[name] == "1" and f"`{name}`" in NOTE


def test_the_note_lists_the_placeholders_of_the_pseudonymiser():
    for placeholder in PLACEHOLDERS:
        assert f"`{placeholder}`" in NOTE


def test_the_note_lists_the_licences_of_the_allowlist():
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    allowed = next(value for key, value in _walk(models) if key == "licences")
    assert allowed == ["apache-2.0", "mit"]
    assert all(f"`{licence}`" in NOTE for licence in allowed)


def _walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk(value)


def test_the_training_hours_are_those_of_the_run_manifest():
    path = ROOT / "results" / "round3" / "runs" / "tax-booking-qwen3-4b-r3-0.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    assert run["num_gpus"] == 2 and f"{run['gpu_hours']:.1f} GPU hours on two T4 GPUs" in NOTE


def test_the_readme_links_the_note_and_shows_the_badge_of_the_workflow_that_exists():
    assert "`docs/compliance.md`" in README
    badge = "actions/workflows/ci.yml/badge.svg?branch=main"
    assert badge in README and (ROOT / ".github" / "workflows" / "ci.yml").is_file()
    assert README.index(badge) < README.index("## What it is for")
