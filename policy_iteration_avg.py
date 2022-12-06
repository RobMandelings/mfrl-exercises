import typing

import numpy as np


def compute_stationary_matrix(transition_matrix: np.matrix) -> np.matrix:
    approximate = np.matrix(np.identity(transition_matrix.size))
    for i in range(10000):
        approximate = np.matmul(approximate, transition_matrix)

    return approximate


def compute_deviation_matrix(fundamental_matrix: np.matrix, stationary_matrix: np.matrix) -> np.matrix:
    return fundamental_matrix - stationary_matrix


def compute_fundamental_matrix(transition_matrix: np.matrix, stationary_matrix: np.matrix) -> np.matrix:
    fundamental_matrix = np.linalg.inv(np.identity(transition_matrix.size) - transition_matrix + stationary_matrix)
    return fundamental_matrix


def compute_avg_reward(stationary_matrix: np.matrix, reward_vector: np.array) -> np.array:
    """
    :return: 1-D array of average rewards for each state
    """
    return np.dot(stationary_matrix, reward_vector)


def compute_u_0(deviation_matrix, reward_matrix) -> np.array:
    """
    :return: 1-D array u_0
    """
    return np.dot(deviation_matrix, reward_matrix)


def compute_B(nr_states: int, nr_actions: int, markov_props: dict,
              reward_matrix: dict, avg_reward: np.array, u_0: np.array) \
        -> typing.Dict[typing.Set]:
    result = {i: set() for i in range(nr_states)}
    for i in range(nr_states):
        for a in range(nr_actions):
            new_avg_reward_inter = map(lambda j: markov_props[i][j][a] * avg_reward[j], list(range(nr_states)))
            # TODO replace with functools?
            new_avg_reward = sum(new_avg_reward_inter)

            add_action = False
            if new_avg_reward > avg_reward[i]:
                add_action = True
            elif new_avg_reward == avg_reward[i]:
                new_u_0_inter = map(lambda j: markov_props[i][j][a] * u_0[j], list(range(nr_states)))
                new_u_0 = sum(new_u_0_inter)
                if reward_matrix[i][a] + new_u_0 > avg_reward[i] + u_0[i]:
                    add_action = True

            if add_action:
                result[i].add(a)

    return result


def create_policy(alpha, f, markov_props, reward_matrix, nr_states, nr_actions):
    pass
