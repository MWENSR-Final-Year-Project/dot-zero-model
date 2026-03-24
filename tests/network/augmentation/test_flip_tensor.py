import numpy as np
import pytest
from network.augmentation import (
    augment_sample,
    flip_lr,
    flip_tensor_lr,
    planes_to_policy,
    policy_to_planes,
    rotate90,
    rotate_tensor,
)

N = 3
POLICY_SIZE = 2 * N * (N + 1)


@pytest.fixture
def random_policy():
    rng = np.random.default_rng(42)
    p = rng.random(POLICY_SIZE).astype(np.float32)
    return p / p.sum()


@pytest.fixture
def random_tensor():
    rng = np.random.default_rng(42)
    return rng.random((N + 1, N + 1, 4)).astype(np.float32)


def test_flip_tensor_lr_shape(random_tensor):
    assert flip_tensor_lr(random_tensor).shape == random_tensor.shape


def test_flip_tensor_lr_returns_copy(random_tensor):
    flipped = flip_tensor_lr(random_tensor)
    flipped[0, 0, 0] = -999.0
    assert random_tensor[0, 0, 0] != -999.0


def test_flip_tensor_lr_twice_is_identity(random_tensor):
    t = flip_tensor_lr(flip_tensor_lr(random_tensor))
    np.testing.assert_array_almost_equal(t, random_tensor)
