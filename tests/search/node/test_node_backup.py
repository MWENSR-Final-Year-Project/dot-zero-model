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
def parent_child(empty_state):
    parent = Node(empty_state, action_size=ACTION_SIZE)
    parent.expand(np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE)
    child = parent.get_child(0)
    return parent, child


def test_backup_updates_parent_visit_count(parent_child):
    parent, child = parent_child
    child.backup(0.5)
    assert parent.child_N[0] == 1.0


def test_backup_updates_parent_value(parent_child):
    parent, child = parent_child
    child.backup(0.5)
    assert parent.child_W[0] == -0.5


def test_backup_updates_total_n(parent_child):
    parent, child = parent_child
    child.backup(0.5)
    assert parent.total_N == 1.0


def test_backup_accumulates_multiple_visits(parent_child):
    parent, child = parent_child
    child.backup(0.5)
    child.backup(0.3)
    assert parent.child_N[0] == 2.0
    assert abs(parent.child_W[0] - (-0.8)) < 1e-6
