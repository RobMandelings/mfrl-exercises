import numpy as np


def compute_value_vector(alpha, stationary_policy, markov_properties, reward_matrix):
    """
    Computes the value vector using the provided data, for a total expected discount problem
    :param alpha: parameter used for infinite horizons
    :param stationary_policy: dictionary with <state, action> key-value pairs
    :param markov_properties: matrix containing all markov properties
    :param reward_matrix: matrix in dictionary form representing the rewards for each action from each state
    :return: value vector for a stationary policy with infinite horizon
    """
    p = create_transition_matrix_for_rule(markov_properties, stationary_policy)
    reward_vector = create_reward_vector_for_rule(reward_matrix, stationary_policy)
    value_vector = np.linalg.inv(np.identity(len(p)) - alpha * p).dot(reward_vector)
    return value_vector


def convert_to_dict(value_vector: np.array):
    """
    Converts the value vector into a numpy array
    :param value_vector:
    :return: dictionary where each start state is a key and each expected reward is a value
    """
    return dict(enumerate(value_vector, 0))


def mult_trans_matrix(matrix1: np.array, matrix2: np.array):
    return np.matmul(matrix1, matrix2)


def create_transition_matrix_for_rule(markov_properties, deterministic_rule) -> np.array:
    transition_matrix = np.zeros((len(markov_properties), len(markov_properties)))

    for i in range(len(markov_properties)):
        # Probability 1, no weighted sum of probabilities required
        chosen_action = deterministic_rule[i]
        for j in range(len(markov_properties)):
            result = markov_properties[i][j][chosen_action]
            transition_matrix[i][j] = result
    return transition_matrix


def calculate_total_expected_reward(nr_states, nr_actions, start_state, markov_props, reward_matrix, policy):
    expected_reward_per_period = []
    res_transition_matrix = np.identity(nr_states)

    for period, rule in policy.items():
        cur_trans_matrix = create_transition_matrix_for_rule(markov_props, rule)
        res_transition_matrix = mult_trans_matrix(res_transition_matrix, cur_trans_matrix)

        transition_probs_from_start = res_transition_matrix[start_state]

        expected_rewards = []
        for j in range(nr_states):
            for a in range(nr_actions):
                # Probability to get to state j with action a, starting from state i
                prob_going_to_j = transition_probs_from_start[j]
                # Reward for going to state j with action a
                prob_a_chosen_at_j = int((rule[j] == a))
                # Probability of going to state j and choosing action a
                prob_going_to_state_j_choosing_a = prob_going_to_j * prob_a_chosen_at_j
                reward_going_state_j_choosing_a = prob_going_to_state_j_choosing_a * reward_matrix[j][a]
                expected_rewards.append(reward_going_state_j_choosing_a)

        expected_reward_per_period.append(sum(expected_rewards))

    return sum(expected_reward_per_period)


def create_reward_vector_for_rule(reward_matrix, deterministic_rule) -> np.array:
    """
    :return: Returns the rewards for the deterministic rule
    """
    reward_vector = np.array([reward_matrix[i][a] for i, a in enumerate(deterministic_rule.values())])
    return reward_vector
