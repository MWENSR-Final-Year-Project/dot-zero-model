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


def _expand_root(puct, state):
    node = puct.select(state)
    puct.expand(node, UNIFORM_POLICY.copy())
    return node


def test_backup_increments_root_total_n(puct, empty_state):
    _expand_root(puct, empty_state)
    child = puct.select(empty_state)
    puct.expand(child, UNIFORM_POLICY.copy())
    puct.backup(child, 0.5)
    assert puct.root.total_N == 1.0


def test_backup_updates_root_child_w(puct, empty_state):
    _expand_root(puct, empty_state)
    child = puct.select(empty_state)
    action = child.action_taken
    puct.expand(child, UNIFORM_POLICY.copy())
    puct.backup(child, 0.6)
    assert abs(puct.root.child_W[action] - (-0.6)) < 1e-6
