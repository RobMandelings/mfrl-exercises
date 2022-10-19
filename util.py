def create_policy(reward_matrix, markov_props, actions, T):
    """
    Creates the optimal policy for this problem using backwards induction
    :param reward_matrix:
    :param markov_props:
    :param T: number of periods
    :return:
    """

    x = [] * (len(T) + 1)
    x[T] = 0

    policy = []

    # Argmax
    for t in reversed(range(T + 1)):
        decision_rule = create_decision_rule(reward_matrix, markov_props, actions, x[t])
        policy.insert(0, decision_rule)  # Insert new rule at the front of list

        if (t - 1) >= 0:
            x[t - 1] = create_x_t(reward_matrix, markov_props, decision_rule, x[t])

    return policy


def create_decision_rule(reward_matrix, markov_props, actions, x_t_plus_1):
    nr_states = len(markov_props)
    decision_rule = [None for _ in range(nr_states)]
    for i in range(nr_states):
        best_result = 0
        for a in actions:
            reward = reward_matrix[i][a]
            weighted_sum = 0
            for j in range(nr_states):
                weighted_sum += (markov_props[i][j][a] * x_t_plus_1[j])

            result = (reward * weighted_sum)
            if result >= best_result:
                best_result = result
                decision_rule[i] = a

    return decision_rule


def create_x_t(reward_matrix, markov_props, decision_rule_f_t, x_t_plus_1):
    reward_vector = create_reward_vector_for_rule(reward_matrix, decision_rule_f_t)
    transition_matrix = create_transition_matrix_for_rule(markov_props, decision_rule_f_t)

    x_t = [0 for _ in range(len(reward_vector))]

    for i in range(len(transition_matrix)):
        result = zip(transition_matrix[i], x_t_plus_1)
        result = map(lambda x: x[0] * x[1], result)
        x_t[i] = sum(result)

    return x_t


def create_reward_vector_for_rule(reward_matrix, deterministic_rule):
    """
    :return: Returns the rewards for the deterministic rule
    """
    reward_vector = {i: reward_matrix[i][a] for i, a in enumerate(deterministic_rule)}
    return reward_vector


def create_transition_matrix_for_rule(markov_properties, deterministic_rule):
    transition_matrix = [[None for _ in range(len(markov_properties))] * len(markov_properties)]

    for i in range(len(markov_properties)):
        # Probability 1, no weighted sum of probabilities required
        chosen_action = deterministic_rule[i]
        for j in range(len(markov_properties)):
            result = markov_properties[i][j][chosen_action]
            transition_matrix[i][j] = result
    return transition_matrix
