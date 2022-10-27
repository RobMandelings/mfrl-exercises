import numpy as np


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


def create_reward_vector_for_rule(reward_matrix, deterministic_rule):
    """
    :return: Returns the rewards for the deterministic rule
    """
    reward_vector = {i: reward_matrix[i][a] for i, a in enumerate(deterministic_rule.values())}
    return reward_vector
