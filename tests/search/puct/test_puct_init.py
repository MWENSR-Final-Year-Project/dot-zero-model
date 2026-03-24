import pytest
from search.puct import PUCT

N = 3
ACTION_SIZE = 2 * N * (N + 1)


@pytest.fixture
def puct():
    return PUCT(action_size=ACTION_SIZE, c_puct=1.5)


def test_init_action_size(puct):
    assert puct.action_size == ACTION_SIZE


def test_init_c_puct(puct):
    assert puct.c_puct == 1.5


def test_init_root_is_none(puct):
    assert puct.root is None
