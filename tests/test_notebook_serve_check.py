import ast
import json
import re
import shutil
import subprocess

import pytest

from adapt.eval.runner import ROOT

NOTEBOOK = ROOT / "notebooks" / "kaggle_serve_check.ipynb"
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code(notebook) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_it_is_a_notebook_with_code_and_text_and_no_saved_output(notebook):
    assert notebook["nbformat"] == 4
    assert {cell["cell_type"] for cell in notebook["cells"]} == {"markdown", "code"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_every_code_cell_is_python(code):
    for source in code:
        compile(source, "<cell>", "exec")


def test_the_notebook_only_defines_the_two_helpers_and_has_no_other_logic(code):
    tree = ast.parse("\n".join(code))
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert functions == {"run", "make"}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def test_every_make_target_it_calls_exists(code):
    targets = set(re.findall(r"^([a-z-]+):", MAKEFILE, re.MULTILINE))
    called = set(re.findall(r'make\(\s*"([a-z-]+)"', "\n".join(code)))
    called |= {"serve-local"} if '"make", "serve-local"' in "\n".join(code) else set()
    assert {"setup", "install-vllm", "fetch-weights", "fetch-adapters", "serve-check"} <= called
    assert called <= targets


def test_the_token_comes_from_a_secret_and_is_never_printed(code):
    source = "\n".join(code)
    assert 'UserSecretsClient().get_secret("HF_TOKEN")' in source
    assert "print(os.environ" not in source and "hf_" not in source


def test_the_keys_are_made_before_the_servers_start_and_the_servers_are_stopped(code):
    source = "\n".join(code)
    assert source.index('"PREPARE=1"') < source.index('"make", "serve-local"')
    assert source.index('"make", "serve-local"') < source.index("DOCUMENTS={DOCUMENTS}")
    assert source.index("DOCUMENTS={DOCUMENTS}") < source.index("os.killpg")


def test_the_repository_is_the_one_of_the_other_notebooks(code):
    train = json.loads((ROOT / "notebooks" / "kaggle_train.ipynb").read_text(encoding="utf-8"))
    url = re.search(r'REPO_URL = "(.*)"', "".join(train["cells"][1]["source"])).group(1)
    assert f'REPO_URL = "{url}"' in code[0]


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_the_make_target_passes_its_variables_on():
    def planned(*variables):
        return subprocess.run(
            ["make", "--dry-run", "--no-print-directory", "serve-check", *variables],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    assert planned().strip().endswith("-m adapt.serve_check")
    assert "--prepare" in planned("PREPARE=1")
    command = planned("DOCUMENTS=3", "WAIT=60", "OUT=x.json")
    assert "--documents 3" in command and "--wait 60" in command and "--out x.json" in command
