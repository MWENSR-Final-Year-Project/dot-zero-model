import numpy as np
import pytest
from game import GameState, Board
from search.puct import PUCT

N = 3
ACTION_SIZE = 2 * N * (N + 1)


@pytest.fixture
def empty_state():
    return GameState(Board(N))


@pytest.fixture
def puct():
    return PUCT(action_size=ACTION_SIZE, c_puct=1.5)


def test_get_policy_temperature1_sums_to_1(puct, empty_state):
    puct.set_root(empty_state)
    puct.root.child_N[0] = 10.0
    puct.root.child_N[1] = 5.0
    p = puct.get_policy(temperature=1.0)
    assert abs(p.sum() - 1.0) < 1e-5


def test_get_policy_temperature0_is_one_hot(puct, empty_state):
    puct.set_root(empty_state)
    puct.root.child_N[3] = 10.0
    puct.root.child_N[0] = 5.0
    p = puct.get_policy(temperature=0)
    assert p[3] == 1.0
    assert p.sum() == 1.0
    assert p[0] == 0.0


def test_get_policy_temperature0_argmax_correct(puct, empty_state):
    puct.set_root(empty_state)
    puct.root.child_N[7] = 100.0
    assert np.argmax(puct.get_policy(temperature=0)) == 7


def test_get_policy_high_temperature_more_uniform(puct, empty_state):
    puct.set_root(empty_state)
    puct.root.child_N[0] = 100.0
    puct.root.child_N[1] = 1.0
    p_low = puct.get_policy(temperature=0.1)
    p_high = puct.get_policy(temperature=2.0)
    assert p_low.max() > p_high.max()


def test_get_policy_all_zeros_returns_zeros(puct, empty_state):
    puct.set_root(empty_state)
    np.testing.assert_array_equal(puct.get_policy(temperature=1.0), 0.0)
