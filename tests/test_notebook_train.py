import ast
import json
import re

import pytest

from adapt.baseline import system_name as baseline_name
from adapt.data.rounds import current_round, eval_dir, runs_dir
from adapt.eval.runner import ROOT
from adapt.models import load_registry
from adapt.train.evaluate import system_name
from adapt.train.run import TrainSettings, run_name

NOTEBOOK = ROOT / "notebooks" / "kaggle_train.ipynb"
MAKEFILE = ROOT / "Makefile"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code(notebook) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def settings_of(source: str) -> dict:
    tree = ast.parse(source)
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
    }


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


def test_a_dry_run_trains_a_tiny_model_and_asks_for_the_fake_server(code):
    text = "\n".join(code)
    assert 'DRY = ["DRY_RUN=1"] if DRY_RUN else []' in text
    assert 'OFFLINE = ["OFFLINE=1"] if DRY_RUN else []' in text
    assert 'SMOKE = ["SMOKE=1"] if DRY_RUN else []' in text
    assert 'make("models-check", *OFFLINE)' in text
    for target, flags in (("train", ("*SMOKE",)), ("eval-sft", ("*DRY", "*SMOKE"))):
        calls = [s for s in code if f'"{target}"' in s]
        assert len(calls) == 2, target  # the short check and the real run
        for call in calls:
            assert all(flag in call for flag in flags), (target, call)


def test_the_adapters_are_saved_right_after_training_and_before_anything_else(code):
    train = next(i for i, s in enumerate(code) if '"train"' in s and "MAX_STEPS" not in s)
    save = code[train + 1]
    assert save.startswith("if DRY_RUN:") and '"zip"' in save
    assert 'f"/kaggle/working/checkpoints-{RUN}.zip"' in save
    assert (
        'f"adapters/tax-booking/{RUN}"' in save
        and 'f"adapters/tax-booking/{RUN}-checkpoints"' in save
    )
    for state in ("optimizer.pt", "scheduler.pt"):
        assert f'"*/{state}"' in save  # the optimiser state is large and of no use afterwards
    assert save.index('run("rm"') < save.index('"zip"')
    assert train + 1 < next(i for i, s in enumerate(code) if '"select"' in s)
    assert train + 1 < next(i for i, s in enumerate(code) if '"eval-sft"' in s and i > train)


def test_the_checkpoint_is_chosen_after_training_and_before_the_adapter_is_evaluated(code):
    select = next(i for i, s in enumerate(code) if '"select"' in s)
    train = next(i for i, s in enumerate(code) if '"train"' in s and "MAX_STEPS" not in s)
    evaluation = next(i for i, s in enumerate(code) if '"eval-sft"' in s and "MAX_STEPS" not in s)
    assert train < select < evaluation
    call = code[select]
    for flag in ("*DRY", "*SMOKE", "*ROUNDS", "*SELECT_STEPS", 'f"SERVERS={SERVERS}"'):
        assert flag in call, flag
    # it reads and writes where the evaluation looks for the adapter, the manifest and the results
    for folder in ("ADAPTERS_DIR", "RUNS_DIR", "OUT_DIR"):
        assert f'f"{folder}={{{folder}.as_posix()}}"' in call, folder


def test_a_dry_run_chooses_between_the_two_checkpoints_of_the_tiny_run(code):
    settings = code[0]
    assert 'SELECT_STEPS = ["STEPS=1 2"] if DRY_RUN else []' in settings


def test_the_notebook_trains_on_two_gpus_but_a_dry_run_has_no_gpu(code):
    settings = settings_of(code[0])
    assert settings["NUM_GPUS"] == 2
    assert 'GPUS = [] if DRY_RUN else [f"NUM_GPUS={NUM_GPUS}"]' in code[0]
    calls = [s for s in code if '"train"' in s]
    assert len(calls) == 2  # the short check and the real run, both on the same GPUs
    assert all("*GPUS" in call for call in calls)
    short_check_evaluation = next(s for s in code if '"eval-sft"' in s and "MAX_STEPS" in s)
    assert "*GPUS" in short_check_evaluation  # it needs the GPU count to find the adapter


def test_the_evaluation_starts_one_server_per_gpu_and_a_dry_run_does_too(code):
    assert settings_of(code[0])["SERVERS"] == 2
    calls = [s for s in code if '"eval-sft"' in s]
    assert len(calls) == 2  # the short check and the real run
    assert all('f"SERVERS={SERVERS}"' in call for call in calls)


def test_a_dry_run_writes_to_temporary_folders_and_a_real_run_inside_the_repository(code):
    cell = next(source for source in code if "def make" in source)
    dry, _, real = cell.partition("else:\n    REPO = ")
    assert "tempfile.mkdtemp" in dry and "/kaggle/working/open-llm-adaptation" in real
    assert "tempfile" not in real


def test_the_notebook_writes_where_the_commands_look_for_the_results_of_its_round(code):
    namespace: dict = {}
    exec(code[0], namespace)
    exec(code[1], namespace)
    repo, round_number = namespace["REPO"], namespace["ROUND"]
    assert namespace["OUT_DIR"].relative_to(repo) == eval_dir(round_number)
    assert namespace["RUNS_DIR"].relative_to(repo) == runs_dir(round_number)


def test_every_make_target_the_notebook_calls_exists(code):
    text = "\n".join(code)
    called = set(re.findall(r'make\(\s*"([a-z-]+)"', text))
    assert called == {
        "setup",
        "install-vllm",
        "models-check",
        "train",
        "select",
        "eval-sft",
        "baseline",
        "gate",
    }
    declared = set(
        re.search(r"^\.PHONY:(.*)$", MAKEFILE.read_text(), re.MULTILINE).group(1).split()
    )
    assert called <= declared


def test_the_model_is_in_the_registry_and_the_check_uses_the_smallest_model(code):
    settings = settings_of(code[0])
    models = load_registry().models
    assert settings["MODEL"] in models
    assert settings["CHECK_MODEL"] == min(models, key=lambda k: models[k].parameters)


def test_the_notebook_trains_on_the_current_round_and_passes_it_to_every_step(code):
    assert settings_of(code[0])["ROUND"] == current_round()
    for target in ("train", "select", "eval-sft", "baseline"):
        calls = [s for s in code if f'"{target}"' in s]
        assert calls, target
        assert all("*ROUNDS" in call for call in calls), target


def test_a_later_round_runs_its_own_baseline_on_test_ood_before_the_gate(code):
    baseline = next(s for s in code if '"baseline"' in s)
    assert baseline.startswith("if FINAL and ROUND > 1:")
    assert '"SPLITS=dev test_ood"' in baseline
    assert code.index(baseline) < next(i for i, s in enumerate(code) if 'make("gate"' in s)


def test_the_notebook_evaluates_dev_and_dev_real_only_until_the_final_run(code):
    assert settings_of(code[0])["FINAL"] is False
    assert 'SPLITS = [] if FINAL else ["SPLITS=dev dev_real"]' in code[0]
    evaluation = next(s for s in code if '"eval-sft"' in s and "MAX_STEPS" not in s)
    assert "*SPLITS" in evaluation
    gate = next(s for s in code if 'make("gate"' in s)
    assert gate.startswith("if FINAL:")


def test_the_last_cell_zips_only_the_files_to_copy_and_is_skipped_in_a_dry_run(code):
    cell = code[-1]
    assert cell.startswith("if not DRY_RUN:")
    assert 'f"results/round{ROUND}/runs/tax-booking-{RUN}.json"' in cell
    assert 'f"results/round{ROUND}/eval/tax-booking/{SYSTEM}"' in cell
    # the scores of the checkpoints are the evidence for the choice in the run manifest
    assert 'glob(f"{SYSTEM}-ckpt*")' in cell
    assert 'f"results/round{ROUND}/eval/tax-booking/{MODEL}-fewshot{TAG}"' in cell
    # the adapter was zipped right after training, so it is not zipped again here
    assert "adapters/" not in cell and "ADAPTER_ZIP" not in cell
    assert cell.count('run("zip"') == 1 and "cwd=REPO" in cell
    # A second run must not add to the archive of the first.
    assert cell.index('run("rm"') < cell.index('run("zip"')
    assert "-steps" not in cell


def test_the_short_check_is_short(code):
    assert 0 < settings_of(code[0])["CHECK_STEPS"] <= 10


@pytest.mark.parametrize("dry_run", [False, True])
def test_the_names_in_the_notebook_are_the_names_the_code_writes(code, dry_run):
    """The notebook lists the adapter and gates the system by name, so they must match."""
    first = code[0].replace("DRY_RUN = False", f"DRY_RUN = {dry_run}", 1)
    namespace: dict = {}
    exec(first, namespace)
    exec(code[1], namespace)
    model, seed = namespace["MODEL"], namespace["SEED"]
    data_round = namespace["ROUND"]
    train = TrainSettings("tax-booking", seed, model=model, smoke=dry_run, data_round=data_round)
    assert namespace["RUN"] == run_name(train)
    assert namespace["SYSTEM"] == system_name(model, seed, smoke=dry_run, data_round=data_round)
    if dry_run:
        assert namespace["BASELINE"] == []
    else:
        assert namespace["BASELINE"] == [f"BASELINE={baseline_name(model, data_round)}"]


def test_the_gate_compares_with_the_baseline_and_may_fail(code):
    gate = next(source for source in code if 'make("gate"' in source)
    assert "check=False" in gate and "*BASELINE" in gate


def test_it_does_not_mention_local_planning_files(notebook):
    assert ".idea" not in json.dumps(notebook)
