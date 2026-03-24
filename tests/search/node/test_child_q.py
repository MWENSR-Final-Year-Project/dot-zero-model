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


def test_child_q_zero_when_unvisited(root_node):
    np.testing.assert_array_equal(root_node.child_Q(), 0.0)


def test_child_q_value(empty_state):
    node = Node(empty_state, action_size=ACTION_SIZE)
    node.child_W[2] = 0.6
    node.child_N[2] = 2.0
    q = node.child_Q()
    assert abs(q[2] - 0.6 / 3.0) < 1e-6
