import numpy as np
import pytest
from network.model import ResidualBlock

N = 3
CHANNELS = 8
BATCH = 2


def test_residual_block_preserves_shape():
    block = ResidualBlock(channels=CHANNELS)
    x = np.random.rand(BATCH, N + 1, N + 1, CHANNELS).astype(np.float32)
    assert block(x).shape == (BATCH, N + 1, N + 1, CHANNELS)


def test_residual_block_output_non_negative():
    block = ResidualBlock(channels=CHANNELS)
    x = np.random.rand(BATCH, N + 1, N + 1, CHANNELS).astype(np.float32)
    assert np.all(block(x).numpy() >= 0)
