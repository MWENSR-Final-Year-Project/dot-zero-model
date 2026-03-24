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

def test_augment_sample_returns_8(random_tensor, random_policy):
    samples = augment_sample(random_tensor, random_policy, 0.5)
    assert len(samples) == 8


def test_augment_sample_tensor_shapes(random_tensor, random_policy):
    for t, _, _ in augment_sample(random_tensor, random_policy, 0.5):
        assert t.shape == (N + 1, N + 1, 4)


def test_augment_sample_policy_shapes(random_tensor, random_policy):
    for _, p, _ in augment_sample(random_tensor, random_policy, 0.5):
        assert p.shape == (POLICY_SIZE,)


def test_augment_sample_value_preserved(random_tensor, random_policy):
    for _, _, v in augment_sample(random_tensor, random_policy, 0.75):
        assert v == 0.75


def test_augment_sample_policy_sums_to_one(random_tensor, random_policy):
    for _, p, _ in augment_sample(random_tensor, random_policy, 0.0):
        assert abs(p.sum() - 1.0) < 1e-5


def test_augment_sample_does_not_mutate_input(random_tensor, random_policy):
    original_tensor = random_tensor.copy()
    original_policy = random_policy.copy()
    augment_sample(random_tensor, random_policy, 0.0)
    np.testing.assert_array_equal(random_tensor, original_tensor)
    np.testing.assert_array_equal(random_policy, original_policy)


def test_augment_sample_all_tensors_are_copies(random_tensor, random_policy):
    samples = augment_sample(random_tensor, random_policy, 0.0)
    for t, _, _ in samples:
        assert t is not random_tensor
