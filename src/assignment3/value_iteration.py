import numpy as np


def create_y_and_f(alpha, x: np.array, nr_states, nr_actions, markov_properties, reward_matrix):
    """

    :param alpha:
    :param x:
    :param nr_states:
    :param nr_actions:
    :param markov_properties:
    :param reward_matrix:
    :return: tuple (f, y) for deterministic policy f and corresponding value vector y
    """
    # For each state, holds an approximation of v_alpha_i
    y = np.empty(shape=nr_states)

    # Returns deterministic policy, approximation of the optimal policy
    f = np.empty(shape=nr_states, dtype=np.int)

    for i in range(nr_states):

        best_result = 0
        best_action = None

        for a in range(nr_actions):
            future_rew_sum = 0
            for j in range(nr_states):
                future_rew_sum += markov_properties[i][j][a] * x[j]
            future_rew_sum *= alpha

            result = reward_matrix[i][a] + future_rew_sum
            if result >= best_result:
                best_result = result
                best_action = a

        y[i] = best_result
        f[i] = best_action

    return f, y


def create_policy(alpha, epsilon, nr_states, nr_actions, markov_properties, reward_matrix):
    """
    Creates the policy using the value-iteration algorithm
    :param alpha:
    :param epsilon:
    :param nr_states:
    :param nr_actions:
    :param markov_properties:
    :param reward_matrix:
    :return:
    """
    bound = (1 - alpha) * epsilon / (2 * alpha)

    x = np.zeros(shape=nr_states)
    f, y = create_y_and_f(alpha, x, nr_states, nr_actions, markov_properties, reward_matrix)

    while np.max(y - x) > bound:
        x = y
        f, y = create_y_and_f(alpha, x, nr_states, nr_actions, markov_properties, reward_matrix)

    return f, y
