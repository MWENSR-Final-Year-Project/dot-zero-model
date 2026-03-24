import numpy as np


def policy_to_planes(policy, board_size):
    """
    Split policy vector into horizontal and vertical edge planes.
    """
    h_size = (board_size + 1) * board_size
    horiz = policy[:h_size].reshape(board_size + 1, board_size)
    vert = policy[h_size:].reshape(board_size, board_size + 1)
    return horiz, vert


def planes_to_policy(horiz, vert):
    """
    Flatten edge planes back into policy vector.
    """
    return np.concatenate([horiz.flatten(), vert.flatten()])


def rotate90(horiz, vert):
    """
    Rotate both planes 90 degrees clockwise.
    """
    new_h = np.rot90(vert, 1)
    new_v = np.rot90(horiz, 1)
    return new_h, new_v


def flip_lr(horiz, vert):
    """
    Horizontal flip.
    """
    return np.fliplr(horiz), np.fliplr(vert)


def rotate_tensor(tensor):
    """
    Rotate HxWxC tensor (state tensor) 90 degrees clockwise.
    Channels remain in same order.
    """
    return np.rot90(tensor, 1, axes=(0, 1)).copy()


def flip_tensor_lr(tensor):
    """
    Horizontal flip of state tensor.
    """
    return np.flip(tensor, axis=1).copy()  # flip W dimension


def augment_sample(state_tensor, policy, value):
    """
    Generate 8 symmetric samples from one board position.
    Returns list of (tensor, policy, value)
    """
    board_size = state_tensor.shape[0] - 1  # H = n+1

    horiz, vert = policy_to_planes(policy, board_size)

    samples = []

    t = state_tensor.copy()

    for _ in range(4):  # rotate 0, 90, 180, 270
        samples.append((t.copy(), planes_to_policy(horiz, vert), value))

        # flipped version
        t_flipped = flip_tensor_lr(t)
        h_f, v_f = flip_lr(horiz, vert)
        samples.append((t_flipped, planes_to_policy(h_f, v_f), value))

        # rotate 90 for next iteration
        horiz, vert = rotate90(horiz, vert)
        t = rotate_tensor(t)

    return samples
