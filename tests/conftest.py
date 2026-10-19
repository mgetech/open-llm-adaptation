import pytest

from fake_server import FakeServer


@pytest.fixture
def server():
    with FakeServer() as fake:
        yield fake


@pytest.fixture(scope="module")
def first_round():
    """Make the first round the current one, for tests of the plain names (`qwen3-4b-0`).

    The real data is in round 2 now, and a round after the first adds `-r2` to the names."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("adapt.data.rounds.current_round", lambda data_dir=None: 1)
        yield
