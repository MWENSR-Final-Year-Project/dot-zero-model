import numpy as np
import pytest
from network.model import DotZeroNet

N = 3
CHANNELS = 8
NUM_RES_BLOCKS = 2
BATCH = 2
ACTION_SIZE = 2 * N * (N + 1)


@pytest.fixture(scope="module")
def model():
    return DotZeroNet(board_size=N, channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS)


@pytest.fixture
def dummy_input():
    rng = np.random.default_rng(0)
    return rng.random((BATCH, N + 1, N + 1, 4)).astype(np.float32)


def test_action_size(model):
    assert model.action_size == ACTION_SIZE


def test_spatial(model):
    assert model.spatial == N + 1


def test_policy_output_shape(model, dummy_input):
    policy, _ = model(dummy_input)
    assert policy.shape == (BATCH, ACTION_SIZE)


def test_value_output_shape(model, dummy_input):
    _, value = model(dummy_input)
    assert value.shape == (BATCH,)


def test_value_bounded_in_minus1_to_1(model, dummy_input):
    _, value = model(dummy_input)
    v = value.numpy()
    assert np.all(v >= -1.0) and np.all(v <= 1.0)


def test_legal_mask_sets_illegal_actions_to_neg_inf(model, dummy_input):
    mask = np.zeros((BATCH, ACTION_SIZE), dtype=np.float32)
    mask[:, 0] = 1.0
    policy, _ = model(dummy_input, legal_mask=mask)
    p = policy.numpy()
    assert np.all(np.isinf(p[:, 1:]) & (p[:, 1:] < 0))
    assert not np.any(np.isinf(p[:, 0]))


def test_legal_mask_does_not_affect_legal_actions(model, dummy_input):
    mask = np.ones((BATCH, ACTION_SIZE), dtype=np.float32)
    policy_masked, _ = model(dummy_input, legal_mask=mask)
    policy_unmasked, _ = model(dummy_input, legal_mask=None)
    np.testing.assert_array_almost_equal(
        policy_masked.numpy(), policy_unmasked.numpy()
    )


def test_no_legal_mask_produces_no_inf(model, dummy_input):
    policy, _ = model(dummy_input, legal_mask=None)
    assert not np.any(np.isinf(policy.numpy()))


def test_call_with_training_true(model, dummy_input):
    policy, value = model(dummy_input, training=True)
    assert policy.shape == (BATCH, ACTION_SIZE)
    assert value.shape == (BATCH,)


@pytest.mark.parametrize("size", [2, 4, 5])
def test_different_board_sizes(size):
    m = DotZeroNet(board_size=size, channels=4, num_res_blocks=1)
    x = np.random.rand(1, size + 1, size + 1, 4).astype(np.float32)
    policy, value = m(x)
    assert policy.shape == (1, 2 * size * (size + 1))
    assert value.shape == (1,)


def test_inference_determinism(model, dummy_input):
    p1, v1 = model(dummy_input, training=False)
    p2, v2 = model(dummy_input, training=False)
    np.testing.assert_array_equal(p1.numpy(), p2.numpy())
    np.testing.assert_array_equal(v1.numpy(), v2.numpy())
