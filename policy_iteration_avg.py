import copy
import random
import typing

import numpy as np

import util


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


def compute_avg_reward_and_u_0(transition_matrix: np.matrix, reward_vector: np.array) -> typing.Tuple[
    np.array, np.array]:
    """
    :return: 1-D array of average rewards for each state
    """

    identity_matrix = np.identity(len(transition_matrix))
    I_P = identity_matrix - transition_matrix

    coefficient_matrix_list = []
    for row in range(len(I_P)):
        coefficient_matrix_list.append([*I_P[row], *np.zeros(shape=64), *np.zeros(shape=64)])
    for row in range(len(I_P)):
        coefficient_matrix_list.append([*identity_matrix[row], *I_P[row], *np.zeros(shape=64)])
    for row in range(len(I_P)):
        coefficient_matrix_list.append([*np.zeros(shape=64), *identity_matrix[row], *I_P[row]])

    # a
    coefficient_matrix = np.array(coefficient_matrix_list)

    dependent_vars_list = []
    for i in range(64):
        dependent_vars_list.append(0)
    for row in range(len(reward_vector)):
        dependent_vars_list.append(reward_vector[row])
    for i in range(64):
        dependent_vars_list.append(0)

    # b
    dependent_variables = np.array(dependent_vars_list)
    result = np.linalg.lstsq(coefficient_matrix, dependent_variables)[0]
    result = np.array(list(map(lambda x: round(x, 10), result)))

    avg_reward = result[:64]
    u_0 = result[64:128]

    return avg_reward, u_0


def compute_B(nr_states: int, nr_actions: int, markov_props: dict,
              reward_matrix: dict, avg_reward: np.array, u_0: np.array, policy) \
        -> typing.Dict[str, typing.Set]:
    result = {i: set() for i in range(nr_states)}

    weird_rewards = {}
    actual_rewards = {}
    for i in range(nr_states):
        for a in range(nr_actions):
            new_avg_reward_inter = map(lambda j: markov_props[i][j][a] * avg_reward[j], list(range(nr_states)))
            new_avg_reward = sum(new_avg_reward_inter)

            # TODO replace with functools?
            add_action = False
            reward_difference = round(new_avg_reward - avg_reward[i], 10)

            if reward_difference > 0:
                add_action = True
                if policy[i] == a:
                    weird_rewards[f">{i},{a}"] = reward_difference
                else:
                    actual_rewards[f">{i},{a}"] = reward_difference
            elif reward_difference == 0:
                new_u_0_inter = map(lambda j: markov_props[i][j][a] * u_0[j], list(range(nr_states)))
                new_u_0 = sum(new_u_0_inter)
                rounded_difference = round((reward_matrix[i][a] + new_u_0) - (avg_reward[i] + u_0[i]), 10)
                if rounded_difference > 0:
                    if policy[i] == a:
                        weird_rewards[f"u0{i},{a}"] = (reward_matrix[i][a] + new_u_0) - (avg_reward[i] + u_0[i])
                    else:
                        actual_rewards[f"u0{i},{a}"] = (reward_matrix[i][a] + new_u_0) - (avg_reward[i] + u_0[i])
                    add_action = True

            if add_action:
                result[i].add(a)

    return result


def create_policy(alpha, f, markov_props, reward_matrix, nr_states, nr_actions):
    """
    :param alpha:
    :param f: initial deterministic policy
    :param markov_props:
    :param reward_matrix:
    :param nr_states:
    :param nr_actions:
    :return:
    """

    optimal_found = False
    avg_reward = None
    policy = copy.deepcopy(f)

    avg_rewards = []

    while not optimal_found:

        transition_matrix = util.create_transition_matrix_for_rule(markov_props, policy)
        reward_vector = util.create_reward_vector_for_rule(reward_matrix, policy)

        result = compute_avg_reward_and_u_0(transition_matrix, reward_vector)
        avg_reward = result[0]
        u_0 = result[1]

        avg_rewards.append(avg_reward)

        B = compute_B(nr_states, nr_actions, markov_props, reward_matrix, avg_reward, u_0, policy)

        total_nr_actions_in_sets = sum(map(lambda actions_set: len(actions_set), B.values()))
        if total_nr_actions_in_sets == 0:
            optimal_found = True
        else:
            g = policy
            for i, actions in B.items():

                if len(actions) > 0:
                    g[i] = list(actions)[0]
            policy = g

    return policy, avg_reward


def test_optimality(markov_props, reward_matrix, nr_actions, policy):
    transition_matrix = util.create_transition_matrix_for_rule(markov_props, policy)
    reward_vector = util.create_reward_vector_for_rule(reward_matrix, policy)
    avg_reward = compute_avg_reward_and_u_0(transition_matrix, reward_vector)[0]

    random_vector = np.zeros(shape=len(policy))
    for i in range(len(policy.values())):
        random_vector[i] = random.randint(0, nr_actions - 1)

    random_vector2 = reward_vector + avg_reward + transition_matrix.dot(random_vector)
    result = random_vector == random_vector2
    return result
