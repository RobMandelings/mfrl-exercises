import gym

# We will load a DiscreteEnv and retrieve the probability and reward
# information
env = gym.make("FrozenLake8x8-v1", desc=None, map_name=None)
"""
DiscreteEnv has an attribute P which holds everything er want as a
dictionary of lists:
P[s][a] = [(probability, nextstate, reward, done), ...]
"""
# first, we initialize the structures with zeros

"""
NOT a transition matrix yet. Simply a matrix that states the probabilities
of going from state i to state j when doing action a.

=> Is not a transition matrix as the probabilty of performing action a 
is not yet included (transition matrices are induced by policies)
"""
markov_properties = {i: {j: {a: 0 for a in range(env.nA)}
                         for j in range(env.nS)}
                     for i in range(env.nS)}

"""
Row (i, j): results in tuple, each element: reward for doing action
"Reward for doing action a while going from state i to state j"
"""
# reward_matrix = {i: {j: {a: 0 for a in range(env.nA)}
#                      for j in range(env.nS)}
#                  for i in range(env.nS)}

# ri_a: reward for action a in state i
reward_matrix = {i: {a: 0 for a in range(env.nA)} for i in range(env.nS)}
# then, we fill them with the actual information
for i in range(env.nS):
    for a in range(env.nA):
        for (p, j, r, d) in env.P[i][a]:
            markov_properties[i][j][a] += p
            reward_matrix[i][a] += r * p

# Policy computation: here's where YOU code
"""
Insert your clever policy computation here! make sure to replace the
policy dictionary below by the results of your computation
"""
T = 2  # Given horizon
policy = {t: {i: env.action_space.sample()
              for i in range(env.nS)}
          for t in range(T)}

# list(map(lambda x: sum(x.values()), list(prob[i].values())))
# Probabilities to get from state i to another state (choosing any action)

# Probability for choosing action 3
# list(map(lambda x: x[3], list(prob[62].values())))


# Policy evaluation: here's where YOU also code
"""
Insert here your code to evaluate
the total expected rewards over the planning horizon T
if one follows your policy. Do the same for a random policy (i.e. the
sample policy given above). As a sanity check, your policy should get an
expected reward of at least the one obtained by the random policy!
"""

# Simulation: you can try your policy here, just remove the false conditional
if True:
    state = env.reset()
    for t in range(T):
        env.render()
        action = policy[t][state]
        print(f"Action = {action}")
        state, reward, done, _ = env.step(action)
        # if the MDP is stuck, we end the simulation here
        if done:
            print(f"Episode finished after {t + 1} timesteps")
            break
    env.close()


def create_policy(reward_matrix, prob_matrix):
    """
    Creates the optimal policy for this problem using backwards induction
    :param reward_matrix:
    :param prob_matrix:
    :return:
    """

    pass


def create_x_t(decision_rule_f_t, x_t_plus_1):
    reward_vector = create_reward_vector_for_rule(reward_matrix, decision_rule_f_t)
    transition_matrix = create_transition_matrix_for_rule(markov_properties, decision_rule_f_t)
    pass


# Creates f_t using the formula
def create_decision_rule(reward_matrix, markov_properties, x_t_plus_1):
    """
    Creates a decision rule based using backwards induction
    :param reward_matrix:
    :param markov_properties:
    :param x_t_plus_1:
    :return:
    """

    decision_rule = list()

    for i in range(len(env.nS)):
        chosen_action = None
        best_result = 0
        for a in range(len(env.nA)):
            result = reward_matrix[i][a]

            for j in range(len(env.nS)):
                result += markov_properties[i][j][a] * x_t_plus_1[j]

            if result >= best_result:
                chosen_action = a

        assert chosen_action is not None, "There should always be a chosen action!"
        decision_rule.append(chosen_action)

    return decision_rule


def create_reward_vector_for_rule(reward_matrix, deterministic_rule):
    """
    :return: Returns the rewards for the deterministic rule
    """
    reward_vector = {i: reward_matrix[i][a] for i, a in enumerate(deterministic_rule)}
    return reward_vector


def create_transition_matrix_for_rule(markov_properties, deterministic_rule):
    pass
