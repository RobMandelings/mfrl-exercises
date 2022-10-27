"""
Policy creation algorithm for the infinite horizon problem
"""

import functools

import numpy as np

import util


def create_policy(alpha, f, markov_properties, reward_matrix, nr_states, nr_actions):
    """

    :param alpha:
    :param f: the start policy (stationary, simply a dict with each state as key and each action as value
    :param markov_properties:
    :param reward_matrix:
    :return:
    """

    optimal_found = False

    while not optimal_found:
        p = util.create_transition_matrix_for_rule(markov_properties, f)
        reward_vector = np.asarray(list(util.create_reward_vector_for_rule(reward_matrix, f).values()))
        value_vector = np.linalg.inv(np.identity(len(p)) - alpha * p).dot(reward_vector)

        s_f_dict = {i: {} for i in range(nr_states)}

        for i in range(nr_states):
            for a in range(nr_actions):
                s_ia_f = reward_matrix[i][a] + alpha * functools.reduce(
                    lambda agg, j: agg + (markov_properties[i][j][a] * value_vector[j]),
                    range(nr_states)) - value_vector[i]
                s_f_dict[i][a] = s_ia_f

        improving_actions = {i: set() for i in range(nr_states)}

        for i in range(nr_states):
            for a in range(nr_actions):
                if s_f_dict[i][a] > 1e-9:
                    improving_actions[i].add(a)

        total_improving_actions = functools.reduce(lambda agg, actions_set: agg + len(actions_set),
                                                   improving_actions.values(), 0)

        if total_improving_actions > 0:
            # Start with g = f
            g = {i: a for i, a in f.items()}

            # Adjust g with improving actions
            for i, actions in improving_actions.items():
                if len(actions) > 0:
                    g[i] = list(actions)[0]

            f = g
        else:
            optimal_found = True

    return f
