"""The container files: what they say, that they follow configs/serve.yaml, and that only the
gateway is reachable. Nothing here needs Docker, except one check that compose accepts the file."""

import shutil
import subprocess

import pytest
import yaml

from adapt.eval.runner import ROOT
from adapt.models import load_registry
from adapt.serve import (
    container_environment,
    container_paths,
    load_serve_config,
    main,
    serve_command,
    vllm_arguments,
)

DEPLOY = ROOT / "deploy"
COMPOSE = DEPLOY / "compose.yaml"
ENV_FILE = DEPLOY / "vllm.env"
DOCKERFILE = DEPLOY / "gateway.Dockerfile"


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_vllm_runs_the_official_image_at_the_pinned_version(compose, registry):
    pinned = registry.gates.vllm.version
    assert compose["services"]["vllm"]["image"] == f"vllm/vllm-openai:v{pinned}"


def test_vllm_is_on_a_network_with_no_way_out_and_publishes_nothing(compose):
    backend = compose["networks"]["backend"]
    vllm = compose["services"]["vllm"]
    assert backend["internal"] is True
    assert vllm["networks"] == ["backend"]
    assert "ports" not in vllm and "expose" not in vllm


def test_only_the_gateway_is_published_and_it_reaches_vllm(compose):
    gateway = compose["services"]["gateway"]
    published = [name for name, s in compose["services"].items() if "ports" in s]
    assert published == ["gateway"] and gateway["ports"] == ["8080:8080"]
    assert set(gateway["networks"]) == {"backend", "frontend"}
    assert not (compose["networks"]["frontend"] or {}).get("internal")


def test_the_gateway_finds_vllm_by_its_service_name_and_the_port_of_the_file(compose):
    port = load_serve_config().port
    assert compose["services"]["gateway"]["environment"]["VLLM_URL"] == f"http://vllm:{port}/v1"


def test_the_gateway_waits_until_vllm_is_healthy(compose):
    assert compose["services"]["gateway"]["depends_on"]["vllm"]["condition"] == "service_healthy"
    assert "healthcheck" in compose["services"]["vllm"]


def test_vllm_reserves_one_gpu(compose):
    (device,) = compose["services"]["vllm"]["deploy"]["resources"]["reservations"]["devices"]
    assert device["driver"] == "nvidia" and device["count"] == 1
    assert "gpu" in device["capabilities"]


def test_the_weights_the_adapters_and_the_keys_are_mounted_read_only(compose):
    mounts = compose["services"]["vllm"]["volumes"] + compose["services"]["gateway"]["volumes"]
    for mount in mounts:
        if "logs" not in mount:
            assert mount.endswith(":ro"), mount
    assert any(m.endswith("/adapters:ro") for m in compose["services"]["vllm"]["volumes"])
    assert any(m.endswith(":/app/adapters:ro") for m in compose["services"]["gateway"]["volumes"])


def test_the_adapters_are_where_the_serve_file_says_in_both_containers(compose):
    config = load_serve_config()
    for adapter in config.adapters:
        assert container_paths(config)[adapter.name] == "/" + adapter.path.as_posix()
        assert adapter.path.parts[0] == "adapters"


def test_the_env_file_is_what_the_serve_file_gives(registry):
    text = container_environment(load_serve_config(), registry)
    assert ENV_FILE.read_bytes() == text.encode("utf-8")
    assert b"\r" not in ENV_FILE.read_bytes()


def test_the_container_gets_the_same_flags_as_the_local_launcher(registry):
    config = load_serve_config()
    local = serve_command(config, registry)[4:]
    container = vllm_arguments(config, registry, container_paths(config))
    assert container[:-1] == local[:-1]
    assert local[-1].startswith("tax-booking=") and container[-1].startswith("tax-booking=/")


def test_the_container_listens_on_all_interfaces_and_serves_offline():
    text = ENV_FILE.read_text(encoding="utf-8")
    assert " --host 0.0.0.0" in text
    for line in ("VLLM_NO_USAGE_STATS=1", "DO_NOT_TRACK=1", "HF_HUB_OFFLINE=1"):
        assert line in text.splitlines()


def test_the_entry_point_writes_the_env_file(tmp_path):
    target = tmp_path / "vllm.env"
    assert main(["--compose-env", str(target)]) == 0
    assert target.read_bytes() == ENV_FILE.read_bytes()


def test_the_gateway_image_is_python_312_slim_with_uv_and_a_user_that_is_not_root():
    text = DOCKERFILE.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    assert lines[2].startswith("FROM python:3.12-slim")
    assert any(line.startswith("COPY --from=ghcr.io/astral-sh/uv:") for line in lines)
    assert "uv sync --frozen --no-dev" in text
    assert any(line.startswith("USER ") and "root" not in line for line in lines)


def test_the_gateway_image_listens_on_the_port_the_compose_file_publishes():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "EXPOSE 8080" in text and '"--port", "8080"' in text and '"0.0.0.0"' in text


def test_neither_the_keys_nor_the_data_go_into_the_image():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "keys.yaml" not in text and "COPY data" not in text and "COPY adapters" not in text
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for path in ("configs/keys.yaml", "adapters", "weights", "logs", "data", "results", "tests"):
        assert path in ignored
    assert "platforms/*/evals" in ignored


def test_every_file_the_dockerfile_copies_exists():
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY ") and "--from" not in line:
            *sources, _ = line.split()[1:]
            for source in sources:
                assert (ROOT / source).exists(), source


needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
needs_compose = pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "compose", "version"], capture_output=True, check=False).returncode
    != 0,
    reason="docker compose is not installed",
)


@needs_compose
def test_docker_compose_accepts_the_file():
    result = subprocess.run(
        ["docker", "compose", "--env-file", str(ENV_FILE), "-f", str(COMPOSE), "config", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@needs_make
def test_the_make_targets_call_docker_and_the_launcher():
    def planned(target, *variables):
        return subprocess.run(
            ["make", "--dry-run", target, *variables],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    assert "-m adapt.serve --compose-env deploy/vllm.env" in planned("compose-env")
    assert "docker compose --env-file deploy/vllm.env -f deploy/compose.yaml config" in planned(
        "compose-config"
    )
    assert "up --build -d" in planned("compose-up")
    assert "--weights-dir weights" in planned("fetch-weights", "WEIGHTS_DIR=weights")
    assert "--weights-dir" not in planned("fetch-weights")
