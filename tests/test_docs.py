"""The architecture document and the decision records agree with the code."""

import re
from typing import get_args

import pytest

from adapt.eval.runner import ROOT
from adapt.models import load_registry
from adapt.pipeline import ReviewReason
from adapt.serve import load_serve_config

DOCS = ROOT / "docs"
ARCHITECTURE = DOCS / "architecture.md"
README = ROOT / "README.md"
RECORDS = sorted((DOCS / "adr").glob("*.md"))
PARTS = ("Status:", "Context:", "Decision:", "Alternatives:", "Consequences:")


def flatten(annotation) -> set[str]:
    arguments = get_args(annotation)
    if not arguments:
        return {annotation}
    return set().union(*(flatten(argument) for argument in arguments))


def diagrams(text: str) -> list[str]:
    return re.findall(r"```mermaid\n(.*?)```", text, re.DOTALL)


def test_there_are_nine_records_numbered_without_gaps():
    assert [path.name[:4] for path in RECORDS] == [f"{n:04d}" for n in range(1, 10)]


@pytest.mark.parametrize("path", RECORDS, ids=lambda path: path.name)
def test_a_record_has_the_five_parts_and_at_most_15_lines(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 15
    assert lines[0].startswith(f"# {path.name[:4]} ")
    text = "\n".join(lines)
    positions = [text.index(part) for part in PARTS]
    assert positions == sorted(positions)


def test_the_pipeline_diagram_names_only_reasons_the_pipeline_can_give():
    reasons = flatten(ReviewReason)
    pipeline = next(d for d in diagrams(ARCHITECTURE.read_text(encoding="utf-8")) if "Pseud" in d)
    named = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", pipeline))
    assert named == reasons


def test_the_document_gives_the_threshold_and_the_routes_of_the_code():
    text = ARCHITECTURE.read_text(encoding="utf-8")
    (adapter,) = load_serve_config().adapters
    assert f"{adapter.threshold:.4f}" in text
    app = (ROOT / "src" / "adapt" / "gateway" / "app.py").read_text(encoding="utf-8")
    for route in ("/v1/{platform}/process", "/healthz", "/readyz", "/metrics"):
        assert route in text and route in app


def test_every_diagram_is_closed_and_the_readme_has_one():
    for path in (ARCHITECTURE, README):
        assert path.read_text(encoding="utf-8").count("```") % 2 == 0
    assert len(diagrams(README.read_text(encoding="utf-8"))) == 1
    assert len(diagrams(ARCHITECTURE.read_text(encoding="utf-8"))) == 2


def test_no_file_says_the_evaluation_sets_were_written_by_hand_or_taken_from_real_invoices():
    phrases = [
        "hand" + "-written",
        "hand" + "written",
        "written by " + "hand",
        "real " + "invoices",
    ]
    skipped = {".venv", ".git", ".idea", "data", "node_modules", "__pycache__", ".ruff_cache"}
    suffixes = {".py", ".md", ".yaml", ".yml", ".txt", ".toml", ".ipynb"}
    found = []
    for path in ROOT.rglob("*"):
        parts = set(path.relative_to(ROOT).parts)
        if not path.is_file() or parts & skipped or path.suffix not in suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        found += [f"{path.relative_to(ROOT)}: {phrase}" for phrase in phrases if phrase in text]
    assert found == []


def test_the_files_the_readme_and_the_document_name_exist():
    pattern = re.compile(r"`((?:docs|deploy|platforms|src|results)/[\w./-]+)`")
    for path in (README, ARCHITECTURE):
        for name in pattern.findall(path.read_text(encoding="utf-8")):
            assert (ROOT / name).exists(), f"{path.name} names {name}"


RUNBOOK = ROOT / "deploy" / "README.md"


def test_the_runbook_only_uses_make_targets_that_exist():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-z-]+):", makefile, re.MULTILINE))
    blocks = re.findall(r"```\n(.*?)```", RUNBOOK.read_text(encoding="utf-8"), re.DOTALL)
    used = set(re.findall(r"^make ([a-z-]+)", "\n".join(blocks), re.MULTILINE))
    assert {
        "setup",
        "fetch-weights",
        "fetch-adapters",
        "compose-up",
        "serve-check",
        "compare",
    } <= used
    assert used <= targets


def test_the_runbook_evaluates_the_adapter_and_the_revision_of_the_serve_file():
    text = RUNBOOK.read_text(encoding="utf-8")
    config = load_serve_config()
    (adapter,) = config.adapters
    assert f"--threshold {adapter.threshold}" in text
    assert f"--adapter-path {adapter.path.as_posix()}" in text
    assert f"--model-revision {load_registry().models[config.model].revision}" in text
    assert f"--model {adapter.name}" in text and f"http://vllm:{config.port}/v1" in text


def test_the_runbook_names_the_kaggle_files_it_compares_with_and_they_exist():
    text = RUNBOOK.read_text(encoding="utf-8")
    names = re.findall(r"FIRST=(\S+)", text)
    assert len(names) == 2
    for name in names:
        assert (ROOT / name).is_file(), name


def test_the_runbook_runs_in_the_planned_region_on_the_planned_size():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "germanywestcentral" in text and "Standard_NC4as_T4_v3" in text
    assert "NO_VLLM=1" in text and "chmod a+rwx logs" in text
    assert text.index("chmod a+rwx logs") < text.index("make compose-up")
    assert text.index("make serve-check PREPARE=1") < text.index("make compose-up")
