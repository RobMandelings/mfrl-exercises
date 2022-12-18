import gym.wrappers
import numpy as np
import numpy.random


def get_action_for_state(Q: np.array, state):
    return np.argmax(Q[state, :])


def get_policy_for_Q(Q: np.array) -> np.array:
    policy = np.zeros(shape=Q.shape[0], dtype=int)
    for state in range(len(Q)):
        policy[state] = get_action_for_state(Q, state)
    return policy


def create_policy(env: gym.wrappers.TimeLimit, gamma, alpha, max_iterations):
    # learning rates => single learning rate?

    Q = np.empty((env.observation_space.n, env.action_space.n))
    initializer = np.vectorize(lambda x: numpy.random.uniform(10, 20))
    Q = initializer(Q)

    state = env.reset()
    iteration = 0
    while iteration < max_iterations:
        action = get_action_for_state(Q, state)
        next_state, reward, done, info = env.step(action)

        Q[state, action] = (1 - gamma) * Q[state, action] + \
                           gamma * (reward + alpha * np.max(Q[next_state, :]))

        state = next_state
        iteration += 1

    env.reset()
    return get_policy_for_Q(Q)

    # TODO check: is assumption that
