from setup import *


class LinearModel(object):

    def __init__(self, number_of_features, number_of_actions):
        self._T = np.zeros((number_of_actions, number_of_features, number_of_features))
        self._R = np.zeros((number_of_actions, number_of_features))
        self._G = np.zeros((number_of_actions, number_of_features))

    def next_state(self, s, a):
        return np.matmul(self._T[a], s)

    def reward(self, s, a):
        return np.matmul(self._R[a], s)

    def discount(self, s, a):
        return np.matmul(self._G[a], s)

    def transition(self, state, action):
        return (
            self.reward(state, action),
            self.discount(state, action),
            self.next_state(state, action))

    def update(self, state, action, reward, discount, next_state, step_size=0.1):
        last_reward, last_discount, last_state = self.transition(state, action)
        temp_state = np.reshape(next_state - last_state, [1, -1])
        self._T[action] += step_size * np.outer(temp_state, state)
        self._R[action] += step_size * (reward - last_reward) * state
        self._G[action] += step_size * (discount - last_discount) * state