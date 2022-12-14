import gym
import numpy as np

import policy_iteration
import policy_iteration_avg

env = gym.make("FrozenLake8x8-v1", desc=None, map_name=None, is_slippery=True)
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
markov_props = {i: {j: {a: 0 for a in range(env.action_space.n)}
                    for j in range(env.observation_space.n)}
                for i in range(env.observation_space.n)}

"""
Row (i, j): results in tuple, each element: reward for doing action
"Reward for doing action a while going from state i to state j"
"""

# ri_a: reward for action a in state i
reward_matrix = {i: {a: 0 for a in range(env.action_space.n)} for i in range(env.observation_space.n)}
# then, we fill them with the actual information
for i in range(env.observation_space.n):
    for a in range(env.action_space.n):
        for (p, j, r, d) in env.P[i][a]:
            markov_props[i][j][a] += p
            reward_matrix[i][a] += r * p

# Policy computation: here's where YOU code
"""
Insert your clever policy computation here! make sure to replace the
policy dictionary below by the results of your computation
"""
T = 1000  # Given horizon
policy_random = {t: {i: env.action_space.sample()
                     for i in range(env.observation_space.n)}
                 for t in range(T)}

# Policy evaluation: here's where YOU also code
"""
Insert here your code to evaluate
the total expected rewards over the planning horizon T
if one follows your policy. Do the same for a random policy (i.e. the
sample policy given above). As a sanity check, your policy should get an
expected reward of at least the one obtained by the random policy!
"""

random_decision_rule = policy_random[0]
# Policy and average reward is computed with policy iteration here
avg_policy, avg_reward = policy_iteration_avg.create_avg_policy(random_decision_rule, markov_props, reward_matrix,
                                                                env.observation_space.n,
                                                                env.action_space.n)

# Average reward is computed for the random policy
avg_reward_random = policy_iteration_avg.compute_avg_reward_and_u_0(reward_matrix, markov_props, random_decision_rule)[
    0]

# Average reward is rounded up to 9 decimals
avg_reward_random = np.array(list(map(lambda x: round(x, 9), avg_reward_random)))

# Both are zero, so the average reward is as least as good as the random average reward
reward_difference = avg_reward - avg_reward_random
for i in range(len(reward_difference)):
    assert reward_difference[
               i] >= 0, f"Reward for {i} of optimal is not bigger than random policy ({avg_reward[i]} not > {avg_reward_random[i]})"

# Compute discounted policy, which is similar to the avg_policy for alpha close to 1
alpha = 0.999
discounted_policy, value_vector = policy_iteration.create_discounted_policy(alpha, random_decision_rule, markov_props,
                                                                            reward_matrix,
                                                                            env.observation_space.n,
                                                                            env.action_space.n)

# TODO compute average reward of random policy using theorem 1.4.7 (item 1)
# TODO output the policy for this assignment
# TODO You print or assert in the code that the average-reward value vector
# TODO (that is, \phi) of your policy is at least that of the random policy.

# Simulation: you can try your policy here, just remove the false conditional
if True:
    state = env.reset()
    for t in range(T):
        env.render()
        # policy is not dependent on time (stationary)
        action = avg_policy[state]
        print(f"Action = {action}")
        state, reward, done, _ = env.step(action)
        # if the MDP is stuck, we end the simulation here
        if done:
            env.render()
            print(f"Episode finished after {t + 1} timesteps")
            break
    env.close()

policy_differences = {}
for i in range(env.observation_space.n):
    policy_differences[i] = (avg_policy[i] != discounted_policy[i])
print(f'Average reward for the optimal policy: {avg_reward}')
print(
    f'Policy using policy iteration with average reward: {list(avg_policy.items())}')
print(
    f'Policy with discounted reward: {list(discounted_policy.items())}')
print(
    f'Differences between discounted policy and the policy with average reward (true if there is a difference, false if they are the same). '
    f'You can see the average policy is similar to the discounted policy')
print(list(policy_differences.items()))
