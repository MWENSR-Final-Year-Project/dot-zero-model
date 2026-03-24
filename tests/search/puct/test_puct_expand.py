import numpy as np
import pytest
from game import GameState, Board
from search.puct import PUCT

N = 3
ACTION_SIZE = 2 * N * (N + 1)
UNIFORM_POLICY = np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE


@pytest.fixture
def empty_state():
    return GameState(Board(N))


@pytest.fixture
def puct():
    return PUCT(action_size=ACTION_SIZE, c_puct=1.5)


def test_expand_marks_node_expanded(puct, empty_state):
    node = puct.select(empty_state)
    puct.expand(node, UNIFORM_POLICY.copy())
    assert node.is_expanded
