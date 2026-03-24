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


def test_rotate90_horiz_shape(random_policy):
    h, v = policy_to_planes(random_policy, N)
    new_h, _ = rotate90(h, v)
    assert new_h.shape == (N + 1, N)


def test_rotate90_vert_shape(random_policy):
    h, v = policy_to_planes(random_policy, N)
    _, new_v = rotate90(h, v)
    assert new_v.shape == (N, N + 1)


def test_rotate90_four_times_is_identity(random_policy):
    h, v = policy_to_planes(random_policy, N)
    original_policy = planes_to_policy(h, v)
    for _ in range(4):
        h, v = rotate90(h, v)
    recovered = planes_to_policy(h, v)
    np.testing.assert_array_almost_equal(recovered, original_policy)
