import numpy as np
import pytest
from game import GameState, Board
from search.node import Node

N = 3
ACTION_SIZE = 2 * N * (N + 1)


@pytest.fixture
def empty_state():
    return GameState(Board(N))


@pytest.fixture
def root_node(empty_state):
    return Node(empty_state, action_size=ACTION_SIZE)


def test_expand_sets_is_expanded(root_node):
    root_node.expand(np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE)
    assert root_node.is_expanded


def test_expand_normalizes_policy(root_node):
    root_node.expand(np.ones(ACTION_SIZE, dtype=np.float32) * 5)
    assert abs(root_node.child_P.sum() - 1.0) < 1e-5


def test_expand_masks_illegal_moves():
    state = GameState(Board(N))
    state.board.horizontal_edges[0, 0] = 1
    node = Node(state, action_size=ACTION_SIZE)
    node.expand(np.ones(ACTION_SIZE, dtype=np.float32))
    assert node.child_P[0] == 0.0


def test_expand_zero_policy_gives_uniform_over_legal(root_node):
    root_node.expand(np.zeros(ACTION_SIZE, dtype=np.float32))
    expected = 1.0 / len(root_node.legal_indices)
    for idx in root_node.legal_indices:
        assert abs(root_node.child_P[idx] - expected) < 1e-6


def test_expand_zero_policy_illegal_stay_zero():
    state = GameState(Board(N))
    state.board.horizontal_edges[0, 0] = 1
    node = Node(state, action_size=ACTION_SIZE)
    node.expand(np.zeros(ACTION_SIZE, dtype=np.float32))
    assert node.child_P[0] == 0.0
