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


def test_select_action_returns_legal_move(root_node):
    action = root_node.select_action(c_puct=1.5)
    assert root_node.legal_mask[action] == 1.0


def test_select_action_picks_highest_prior(empty_state):
    node = Node(empty_state, action_size=ACTION_SIZE)
    node.child_P[7] = 1.0
    assert node.select_action(c_puct=1.5) == 7


def test_select_action_avoids_illegal_moves():
    state = GameState(Board(N))
    state.board.horizontal_edges[0, 0] = 1
    node = Node(state, action_size=ACTION_SIZE)
    node.child_P[:] = 1.0 / ACTION_SIZE
    assert node.select_action(c_puct=1.5) != 0
