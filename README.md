# DotZero Model

AlphaZero-style neural network and MCTS search for the Dots-and-Boxes game. Built with TensorFlow/Keras and designed to train on Google Cloud TPUs via GKE.

## Overview

DotZero uses self-play reinforcement learning to master Dots-and-Boxes. At each turn, a PUCT search tree is built by running simulations guided by a dual-head convolutional neural network. The network outputs:

- **Policy head** — a probability distribution over all possible edge placements
- **Value head** — an estimate of the current player's winning probability in [-1, 1]

After training, the network improves the search, and the search generates better training data — the AlphaZero feedback loop.

## Architecture

```
Input (H, W, C) = (n+1, n+1, 4)  — channels-last for TPU compatibility
    │
    ├── Channel 0: horizontal edges
    ├── Channel 1: vertical edges
    ├── Channel 2: current player's completed boxes
    └── Channel 3: opponent's completed boxes

    ↓
Initial Conv + BatchNorm + ReLU
    ↓
Residual Tower (6 × ResidualBlock)
    ↓
    ├── Policy Head → Conv(2, 1×1) → BN → flatten → legal mask → logits (action_size,)
    └── Value Head  → Conv(1, 1×1) → BN → flatten → Dense(64) → Dense(1) → tanh → scalar
```

Default hyperparameters: `board_size=5`, `channels=64`, `num_res_blocks=6`.

## Project Structure

```
src/
  network/
    model.py        # DotZeroNet and ResidualBlock (TensorFlow/Keras)
    encoding.py     # Game state → numpy tensor (channels-last)
    augmentation.py # 8-fold symmetry augmentation for training data
  search/
    node.py         # MCTS tree node with PUCT statistics
    puct.py         # PUCT search (select, expand, backup, get_policy)

tests/
  network/
    augmentation/   # One test file per augmentation function
    encoding/
    model/          # test_dot_zero_net.py, test_residual_block.py
  search/
    node/           # One test file per Node method
    puct/           # One test file per PUCT method
```

## Requirements

- Python 3.12+
- [dot-zero-game](https://github.com/MWENSR-Final-Year-Project/dot-zero-game) — game engine and state representation
- TensorFlow >= 2.15
- NumPy >= 1.24

Dependencies are declared in `pyproject.toml` and installed automatically.

## Installation

```bash
pip install .
```

For development (includes pytest and pytest-cov):

```bash
pip install ".[dev]"
```

## Running Tests

```bash
pytest
```

With coverage report:

```bash
pytest --cov --cov-report=term-missing
```

## Search Algorithm

The search uses **PUCT** (Polynomial Upper Confidence Trees), the AlphaZero variant of MCTS. Action selection at each node follows:

```
score(a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

where `Q` is the mean value, `P` is the network's policy prior, and `N` is the visit count. Default `c_puct = 1.5`.

### Training vs Inference

| Feature | Training | Inference |
|---|---|---|
| Temperature | 1.0 for first ~30% of moves, then 0 | 0 (greedy) |
| Dirichlet noise | Yes — `puct.add_dirichlet_noise()` at root | No |

Temperature scheduling and Dirichlet noise (ε=0.25, α=10/action_size) ensure the self-play games are diverse enough to generate useful training data.

## Data Augmentation

Dots-and-Boxes is symmetric under 4 rotations and horizontal flip, giving 8 equivalent board positions per game state. `augment_sample(state_tensor, policy, value)` returns all 8, multiplying dataset size by 8 with no extra games needed.
