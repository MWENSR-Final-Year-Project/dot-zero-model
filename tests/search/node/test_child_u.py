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


def test_child_u_zero_when_no_prior(root_node):
    np.testing.assert_array_equal(root_node.child_U(c_puct=1.5), 0.0)


def test_child_u_proportional_to_prior(empty_state):
    node = Node(empty_state, action_size=ACTION_SIZE)
    node.child_P[0] = 0.8
    node.child_P[1] = 0.2
    node.total_N = 4.0
    u = node.child_U(c_puct=1.0)
    assert u[0] > u[1]
