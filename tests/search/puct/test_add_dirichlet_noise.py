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


def test_dirichlet_noise_noop_before_expand(puct, empty_state):
    puct.set_root(empty_state)
    original = puct.root.child_P.copy()
    puct.add_dirichlet_noise()
    np.testing.assert_array_equal(puct.root.child_P, original)


def test_dirichlet_noise_noop_when_root_none(puct):
    puct.add_dirichlet_noise()


def test_dirichlet_noise_policy_sums_to_1(puct, empty_state):
    _expand_root(puct, empty_state)
    puct.add_dirichlet_noise(epsilon=0.25)
    assert abs(puct.root.child_P.sum() - 1.0) < 1e-5


def test_dirichlet_noise_illegal_indices_stay_zero():
    state = GameState(Board(N))
    state.board.horizontal_edges[0, 0] = 1
    p = PUCT(action_size=ACTION_SIZE)
    node = p.select(state)
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[1] = 1.0
    p.expand(node, policy)
    p.add_dirichlet_noise(epsilon=0.25)
    assert p.root.child_P[0] == 0.0


def test_dirichlet_noise_changes_prior(puct, empty_state):
    _expand_root(puct, empty_state)
    before = puct.root.child_P.copy()
    np.random.seed(99)
    puct.add_dirichlet_noise(epsilon=0.25)
    assert not np.array_equal(puct.root.child_P, before)


def test_dirichlet_noise_custom_alpha(puct, empty_state):
    _expand_root(puct, empty_state)
    puct.add_dirichlet_noise(epsilon=0.25, alpha=0.5)
    assert abs(puct.root.child_P.sum() - 1.0) < 1e-5
