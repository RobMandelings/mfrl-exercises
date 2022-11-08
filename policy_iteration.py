"""
Policy creation algorithm for the infinite horizon problem
"""

import functools

import util


def create_policy(alpha, f, markov_properties, reward_matrix, nr_states, nr_actions):
    """
    Creates a stationary deterministic policy using the policy iteration algorithm
    :param alpha: parameter used for infinite horizons
    :param f: the start policy (stationary, simply a dict with each state as key and each action as value
    :param markov_properties: matrix containing all markov properties
    :param reward_matrix: matrix in dictionary form representing the rewards for each action from each state
    :return: tuple (stationary_policy, value_vector (dict form) )
    """

    result = None
    nr_iterations = 0

    while result is None:
        value_vector = util.compute_value_vector(alpha, f, markov_properties, reward_matrix)
        s_f_dict = {i: {} for i in range(nr_states)}

        for i in range(nr_states):
            for a in range(nr_actions):
                summation = 0
                for j in range(nr_states):
                    summation += (markov_properties[i][j][a] * value_vector[j])
                s_ia_f = reward_matrix[i][a] + alpha * summation - value_vector[i]
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
            # Return policy and value vector. Value vector will be returned as a dictionary
            result = f, util.convert_to_dict(value_vector)
        nr_iterations += 1

    return result
