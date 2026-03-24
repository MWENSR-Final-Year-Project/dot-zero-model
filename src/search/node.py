import numpy as np

from game import index_to_move, move_to_index, GameState
from network.encoding import state_to_tensor


class Node:
    def __init__(
        self,
        state: GameState,
        parent=None,
        action_taken=None,
        action_size=None,
    ):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.action_size = action_size

        self.tensor = state_to_tensor(state)
        self.hash = hash(state)

        self.children = {}

        self.child_N = np.zeros(action_size, dtype=np.float32)
        self.child_W = np.zeros(action_size, dtype=np.float32)
        self.child_P = np.zeros(action_size, dtype=np.float32)

        self.total_N = 0.0

        legal_moves = state.get_legal_moves()
        self.legal_indices = np.array(
            [move_to_index(m, state.board.size) for m in legal_moves],
            dtype=np.int32,
        )

        self.legal_mask = np.zeros(action_size, dtype=np.float32)
        self.legal_mask[self.legal_indices] = 1.0

        self.is_expanded = False

    def child_Q(self):
        return self.child_W / (1.0 + self.child_N)

    def child_U(self, c_puct):
        return (
            c_puct * self.child_P * np.sqrt(self.total_N + 1e-8) / (1.0 + self.child_N)
        )

    def select_action(self, c_puct):
        scores = self.child_Q() + self.child_U(c_puct)

        # mask illegal moves
        scores = scores * self.legal_mask - (1 - self.legal_mask) * 1e9

        return int(np.argmax(scores))

    def expand(self, policy):
        self.is_expanded = True

        if len(self.legal_indices) == 0:
            return
        # mask policy once
        policy = policy * self.legal_mask
        policy_sum = policy.sum()

        if policy_sum > 0:
            policy /= policy_sum
        else:
            policy[self.legal_indices] = 1.0 / len(self.legal_indices)

        self.child_P = policy

    def get_child(self, action_index):
        if action_index not in self.children:
            move = index_to_move(action_index, self.state.board.size)

            next_state = self.state.clone().apply_move(move)

            self.children[action_index] = Node(
                next_state,
                parent=self,
                action_taken=action_index,
                action_size=self.action_size,
            )

        return self.children[action_index]

    def backup(self, value):

        node = self

        while node.parent is not None:
            parent = node.parent

            if parent.state.current_player != node.state.current_player:
                value = -value
            parent.child_N[node.action_taken] += 1
            parent.child_W[node.action_taken] += value
            parent.total_N += 1

            node = parent
