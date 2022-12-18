import gym
import numpy as np
import pygame

import q_learning
import util

# print(gym.__version__)

"""
You will not be able to see the simulation in colab. You can do it 
locally by un-commenting the following 4 lines and env.render() in the 
last piece of the code
"""

pygame.init()
pygame.display.init()
pygame.display.list_modes()

# We will load a DiscreteEnv and retrieve the probability and reward
# information
env = gym.make("FrozenLake8x8-v1", desc=None, map_name=None, is_slippery=True)
"""
DiscreteEnv has an attribute P which holds everything we want as a
dictionary of lists:
P[s][a] = [(probability, nextstate, reward, done), ...]
"""

markov_props = np.zeros(shape=(env.observation_space.n,
                               env.observation_space.n,
                               env.action_space.n))

"""
Row (i, j): results in tuple, each element: reward for doing action
"Reward for doing action a while going from state i to state j"
"""

# ri_a: reward for action a in state i
reward_matrix = np.zeros(shape=(env.observation_space.n, env.action_space.n))
# then, we fill them with the actual information
for i in range(env.observation_space.n):
    for a in range(env.action_space.n):
        for (p, j, r, d) in env.P[i][a]:
            markov_props[i, j, a] += p
            reward_matrix[i, a] += r * p

# Policy computation: here's where YOU code
"""
Insert your clever policy computation here! make sure to replace the
policy dictionary below by the results of your computation
"""

alpha = 0.999

q_learning_pol = q_learning.create_policy(env, gamma=0.5, alpha=alpha, max_iterations=100_000)

T = 10  # Given horizon
random_policy = {i: env.action_space.sample()
                 for i in range(env.observation_space.n)}

# Policy evaluation: here's where YOU also code
"""
Insert here your code to evaluate
the total expected rewards over the planning horizon T
if one follows your policy. Do the same for a random policy (i.e. the
sample policy given above). As a sanity check, your policy should get an
expected reward of at least the one obtained by the random policy!
"""

# Simulation: you can try your policy here
state = env.reset()
for i, t in enumerate(range(T)):
    env.render()
    action = q_learning_pol[state]
    print(f"Action = {action}")
    state, reward, done, info = env.step(action)

    # if the MDP is stuck, we end the simulation here
    if done:
        env.render()
        print(f"Episode finished after {t + 1} timesteps")
        break
env.close()

reward_q_learning_pol = util.compute_value_vector(alpha, q_learning_pol, markov_props, reward_matrix)
reward_random_policy = util.compute_value_vector(alpha, random_policy, markov_props, reward_matrix)

assert (reward_q_learning_pol - reward_random_policy) >= 0

print(f"Expected discounted rewards for Q learned policy vs. random policy")
print(f"Q learned policy: {reward_q_learning_pol[0]}")
print(f"Random policy: {reward_random_policy[0]}")
