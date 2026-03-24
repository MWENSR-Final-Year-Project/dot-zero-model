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
    node = Node(empty_state, action_size=ACTION_SIZE)
    node.expand(np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE)
    return node


def test_get_child_returns_node(root_node):
    assert isinstance(root_node.get_child(0), Node)


def test_get_child_sets_parent(root_node):
    assert root_node.get_child(0).parent is root_node


def test_get_child_sets_action_taken(root_node):
    assert root_node.get_child(0).action_taken == 0


def test_get_child_is_cached(root_node):
    assert root_node.get_child(0) is root_node.get_child(0)
