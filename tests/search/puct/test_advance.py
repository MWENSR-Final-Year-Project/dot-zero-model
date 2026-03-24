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


def test_advance_known_child_updates_root(puct, empty_state):
    node = puct.select(empty_state)
    puct.expand(node, UNIFORM_POLICY.copy())
    puct.root.get_child(0)
    old_root = puct.root
    puct.advance(0)
    assert puct.root is not old_root
    assert puct.root.action_taken == 0


def test_advance_known_child_clears_parent(puct, empty_state):
    node = puct.select(empty_state)
    puct.expand(node, UNIFORM_POLICY.copy())
    puct.root.get_child(0)
    puct.advance(0)
    assert puct.root.parent is None


def test_advance_unknown_action_resets_root(puct, empty_state):
    puct.set_root(empty_state)
    puct.advance(ACTION_SIZE - 1)
    assert puct.root is None
