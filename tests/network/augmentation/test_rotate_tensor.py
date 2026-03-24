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


def test_rotate_tensor_shape(random_tensor):
    assert rotate_tensor(random_tensor).shape == random_tensor.shape


def test_rotate_tensor_returns_copy(random_tensor):
    rotated = rotate_tensor(random_tensor)
    rotated[0, 0, 0] = -999.0
    assert random_tensor[0, 0, 0] != -999.0


def test_rotate_tensor_four_times_is_identity(random_tensor):
    t = random_tensor.copy()
    for _ in range(4):
        t = rotate_tensor(t)
    np.testing.assert_array_almost_equal(t, random_tensor)
