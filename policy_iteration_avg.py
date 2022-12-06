import numpy as np


def compute_stationary_matrix() -> np.matrix:
    pass


def compute_deviation_matrix() -> np.matrix:
    pass


def compute_avg_reward(stationary_matrix: np.matrix, reward_vector: np.array) -> np.array:
    """
    :return: 1-D array of average rewards for each state
    """
    pass


def compute_u_0(deviation_matrix, reward_matrix) -> np.array:
    pass


def compute_B_i_f(state: int, nr_actions, avg_reward: np.array, u_0: np.array) -> set:
    for action in range(nr_actions):
        pass


def create_policy(alpha, f, markov_properties, reward_matrix, nr_states, nr_actions):
    pass
