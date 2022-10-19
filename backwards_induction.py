import util


def create_policy(reward_matrix, markov_props, nr_actions, T):
    """
    Creates the optimal policy for this problem using backwards induction
    :param reward_matrix:
    :param markov_props:
    :param T: number of periods
    :return:
    """

    x = {T: [0 for _ in range(len(markov_props))]}

    policy = {}

    # Argmax
    for t in reversed(range(T)):
        decision_rule = create_decision_rule(reward_matrix, markov_props, nr_actions, x[t + 1])
        policy[t] = decision_rule  # Insert new rule at the front of list

        if t > 0:
            x[t] = create_x_t(reward_matrix, markov_props, decision_rule, x[t + 1])

    value_vector = x[1]

    return policy, value_vector


def create_decision_rule(reward_matrix, markov_props, nr_actions, x_t_plus_1):
    nr_states = len(markov_props)
    decision_rule = {}
    for i in range(nr_states):
        best_result = 0
        for a in range(nr_actions):
            reward = reward_matrix[i][a]
            weighted_sum = 0
            for j in range(nr_states):
                weighted_sum += (markov_props[i][j][a] * x_t_plus_1[j])

            result = (reward + weighted_sum)
            if result >= best_result:
                best_result = result
                decision_rule[i] = a

    assert len(decision_rule.keys()) == nr_states
    return decision_rule


def create_x_t(reward_matrix, markov_props, decision_rule_f_t, x_t_plus_1):
    reward_vector = create_reward_vector_for_rule(reward_matrix, decision_rule_f_t)
    transition_matrix = util.create_transition_matrix_for_rule(markov_props, decision_rule_f_t)

    x_t = [reward_vector[i] for i in reward_vector.keys()]

    for i in range(len(transition_matrix)):
        result = zip(list(transition_matrix[i]), x_t_plus_1)
        result = map(lambda x: x[0] * x[1], result)
        x_t[i] += sum(result)

    return x_t


def create_reward_vector_for_rule(reward_matrix, deterministic_rule):
    """
    :return: Returns the rewards for the deterministic rule
    """
    reward_vector = {i: reward_matrix[i][a] for i, a in enumerate(deterministic_rule.values())}
    return reward_vector
