import pytest

from fake_server import FakeServer


@pytest.fixture
def server():
    with FakeServer() as fake:
        yield fake
