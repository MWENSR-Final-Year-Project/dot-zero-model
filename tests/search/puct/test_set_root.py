import pytest
from game import GameState, Board
from search.node import Node
from search.puct import PUCT

N = 3
ACTION_SIZE = 2 * N * (N + 1)


@pytest.fixture
def empty_state():
    return GameState(Board(N))


@pytest.fixture
def puct():
    return PUCT(action_size=ACTION_SIZE, c_puct=1.5)


def test_set_root_creates_node(puct, empty_state):
    puct.set_root(empty_state)
    assert isinstance(puct.root, Node)


def test_set_root_node_not_expanded(puct, empty_state):
    puct.set_root(empty_state)
    assert not puct.root.is_expanded
