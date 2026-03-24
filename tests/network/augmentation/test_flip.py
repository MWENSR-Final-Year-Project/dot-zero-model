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


def test_flip_lr_shapes(random_policy):
    h, v = policy_to_planes(random_policy, N)
    new_h, new_v = flip_lr(h, v)
    assert new_h.shape == h.shape
    assert new_v.shape == v.shape


def test_flip_lr_twice_is_identity(random_policy):
    h, v = policy_to_planes(random_policy, N)
    h2, v2 = flip_lr(h, v)
    h3, v3 = flip_lr(h2, v2)
    np.testing.assert_array_equal(h3, h)
    np.testing.assert_array_equal(v3, v)