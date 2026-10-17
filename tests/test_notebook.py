import ast
import json
import re

import pytest

from adapt.eval.runner import ROOT
from adapt.models import load_registry

NOTEBOOK = ROOT / "notebooks" / "kaggle_baselines.ipynb"
MAKEFILE = ROOT / "Makefile"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code(notebook) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_it_is_a_notebook_with_code_and_text(notebook):
    assert notebook["nbformat"] == 4
    kinds = {cell["cell_type"] for cell in notebook["cells"]}
    assert kinds == {"markdown", "code"}


def test_it_has_no_saved_output(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_every_code_cell_is_python(code):
    for source in code:
        compile(source, "<cell>", "exec")


def test_the_dry_run_flag_is_off_in_the_committed_notebook(code):
    assert code[0].splitlines()[0] == "DRY_RUN = False"


def test_the_notebook_only_defines_the_two_helpers_and_has_no_other_logic(code):
    tree = ast.parse("\n".join(code))
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert functions == {"run", "make"}
    forbidden = (ast.While, ast.Try, ast.ListComp, ast.DictComp, ast.Lambda, ast.With)
    inside_run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    allowed_with = [n for n in ast.walk(inside_run) if isinstance(n, ast.With)]
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and node in allowed_with:
            continue
        assert not isinstance(node, forbidden), ast.dump(node)


def test_the_steps_that_need_a_gpu_or_the_network_are_skipped_in_a_dry_run(code):
    needs_machine = ('run("pip"', 'run("uv"', 'run("git"', 'make("setup")', 'make("install-vllm")')
    cells = [source for source in code if any(call in source for call in needs_machine)]
    assert len(cells) == 3
    for source in cells:
        head, _, tail = source.partition("else:")
        assert head.startswith("if DRY_RUN:") and "run(" not in head and "make(" not in head
        assert any(call in tail for call in needs_machine)


def test_a_dry_run_asks_for_the_fake_server_and_for_no_hub_lookup(code):
    text = "\n".join(code)
    assert 'DRY = ["DRY_RUN=1"] if DRY_RUN else []' in text
    assert 'OFFLINE = ["OFFLINE=1"] if DRY_RUN else []' in text
    assert 'make("baseline", f"MODEL={model}", f"OUT_DIR={OUT_DIR.as_posix()}", *DRY)' in text
    assert 'make("models-check", *OFFLINE)' in text


def test_every_make_target_the_notebook_calls_exists(code):
    text = "\n".join(code)
    called = set(re.findall(r'make\("([a-z-]+)"', text))
    assert called == {"setup", "install-vllm", "models-check", "baseline", "gate"}
    declared = set(
        re.search(r"^\.PHONY:(.*)$", MAKEFILE.read_text(), re.MULTILINE).group(1).split()
    )
    assert called <= declared


def test_the_notebook_runs_the_candidates_of_the_registry(code):
    tree = ast.parse(code[0])
    models = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and n.targets[0].id == "MODELS"
    )
    registry = load_registry()
    candidates = [key for key, entry in registry.models.items() if entry.role == "candidate"]
    assert models == candidates


def test_the_gate_is_expected_to_fail_without_stopping_the_notebook(code):
    gate = next(source for source in code if 'make("gate"' in source)
    assert "check=False" in gate


def test_it_does_not_mention_local_planning_files(notebook):
    assert ".idea" not in json.dumps(notebook)
