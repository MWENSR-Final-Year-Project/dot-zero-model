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

def test_policy_to_planes_horiz_shape(random_policy):
    h, _ = policy_to_planes(random_policy, N)
    assert h.shape == (N + 1, N)


def test_policy_to_planes_vert_shape(random_policy):
    _, v = policy_to_planes(random_policy, N)
    assert v.shape == (N, N + 1)


def test_planes_to_policy_roundtrip(random_policy):
    h, v = policy_to_planes(random_policy, N)
    recovered = planes_to_policy(h, v)
    np.testing.assert_array_almost_equal(recovered, random_policy)


def test_planes_to_policy_length(random_policy):
    h, v = policy_to_planes(random_policy, N)
    p = planes_to_policy(h, v)
    assert len(p) == POLICY_SIZE
