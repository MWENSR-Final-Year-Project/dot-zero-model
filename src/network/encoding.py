import numpy as np
from game import GameState


def state_to_tensor(state: GameState):
    board = state.board
    n = board.size

    H = W = n + 1
    C = 4

    tensor = np.zeros((H, W, C), dtype=np.float32)

    tensor[:, :n, 0] = board.horizontal_edges.astype(np.float32)
    tensor[:n, :, 1] = board.vertical_edges.astype(np.float32)

    boxes = board.boxes

    if state.current_player == 1:
        current_boxes = (boxes == 1).astype(np.float32)
        opponent_boxes = (boxes == -1).astype(np.float32)
    else:
        current_boxes = (boxes == -1).astype(np.float32)
        opponent_boxes = (boxes == 1).astype(np.float32)

    tensor[:n, :n, 2] = current_boxes
    tensor[:n, :n, 3] = opponent_boxes

    return tensor
