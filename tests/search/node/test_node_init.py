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


def test_initial_is_not_expanded(root_node):
    assert not root_node.is_expanded


def test_initial_total_n(root_node):
    assert root_node.total_N == 0.0


def test_initial_child_arrays_zero(root_node):
    np.testing.assert_array_equal(root_node.child_N, 0)
    np.testing.assert_array_equal(root_node.child_W, 0)
    np.testing.assert_array_equal(root_node.child_P, 0)


def test_child_arrays_shape(root_node):
    assert root_node.child_N.shape == (ACTION_SIZE,)
    assert root_node.child_W.shape == (ACTION_SIZE,)
    assert root_node.child_P.shape == (ACTION_SIZE,)


def test_tensor_shape(root_node):
    assert root_node.tensor.shape == (N + 1, N + 1, 4)


def test_legal_mask_all_ones_on_empty_board(root_node):
    assert root_node.legal_mask.sum() == ACTION_SIZE


def test_legal_mask_zeros_filled_edge():
    state = GameState(Board(N))
    state.board.horizontal_edges[0, 0] = 1
    node = Node(state, action_size=ACTION_SIZE)
    assert node.legal_mask[0] == 0.0
    assert node.legal_mask.sum() == ACTION_SIZE - 1
