"""The architecture document and the decision records agree with the code."""

import re
from typing import get_args

import pytest

from adapt.eval.runner import ROOT
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


def test_the_files_the_readme_and_the_document_name_exist():
    pattern = re.compile(r"`((?:docs|deploy|platforms|src|results)/[\w./-]+)`")
    for path in (README, ARCHITECTURE):
        for name in pattern.findall(path.read_text(encoding="utf-8")):
            assert (ROOT / name).exists(), f"{path.name} names {name}"
