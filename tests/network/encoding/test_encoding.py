import numpy as np
import pytest
from game import GameState, Board
from network.encoding import state_to_tensor

N = 3


@pytest.fixture
def empty_state():
    return GameState(Board(N))


def test_output_shape(empty_state):
    t = state_to_tensor(empty_state)
    assert t.shape == (N + 1, N + 1, 4)


def test_dtype(empty_state):
    t = state_to_tensor(empty_state)
    assert t.dtype == np.float32


def test_empty_board_all_zeros(empty_state):
    t = state_to_tensor(empty_state)
    np.testing.assert_array_equal(t, 0)


def test_horizontal_edge_in_channel_0():
    state = GameState(Board(N))
    state.board.horizontal_edges[0, 0] = 1
    t = state_to_tensor(state)
    assert t[0, 0, 0] == 1.0
    assert t[:, :, 1].sum() == 0
    assert t[:, :, 2].sum() == 0
    assert t[:, :, 3].sum() == 0


def test_vertical_edge_in_channel_1():
    state = GameState(Board(N))
    state.board.vertical_edges[0, 0] = 1
    t = state_to_tensor(state)
    assert t[0, 0, 1] == 1.0
    assert t[:, :, 0].sum() == 0


def test_current_player_1_box_in_channel_2():
    state = GameState(Board(N), current_player=1)
    state.board.boxes[0, 0] = 1
    t = state_to_tensor(state)
    assert t[0, 0, 2] == 1.0
    assert t[0, 0, 3] == 0.0


def test_current_player_minus1_box_in_channel_2():
    state = GameState(Board(N), current_player=-1)
    state.board.boxes[0, 0] = -1
    t = state_to_tensor(state)
    assert t[0, 0, 2] == 1.0
    assert t[0, 0, 3] == 0.0


def test_opponent_box_in_channel_3():
    state = GameState(Board(N), current_player=1)
    state.board.boxes[0, 0] = -1
    t = state_to_tensor(state)
    assert t[0, 0, 2] == 0.0
    assert t[0, 0, 3] == 1.0


def test_perspective_swap_between_players():
    # Same physical board, opposite players — channels 2 and 3 should swap.
    state1 = GameState(Board(N), current_player=1)
    state2 = GameState(Board(N), current_player=-1)
    state1.board.boxes[0, 0] = 1
    state2.board.boxes[0, 0] = 1
    t1 = state_to_tensor(state1)
    t2 = state_to_tensor(state2)
    np.testing.assert_array_equal(t1[:, :, 2], t2[:, :, 3])
    np.testing.assert_array_equal(t1[:, :, 3], t2[:, :, 2])


def test_multiple_edges_encoded():
    state = GameState(Board(N))
    state.board.horizontal_edges[0, 0] = 1
    state.board.horizontal_edges[1, 2] = 1
    state.board.vertical_edges[2, 1] = 1
    t = state_to_tensor(state)
    assert t[0, 0, 0] == 1.0
    assert t[1, 2, 0] == 1.0
    assert t[2, 1, 1] == 1.0
    assert t[:, :, 0].sum() == 2.0
    assert t[:, :, 1].sum() == 1.0
