from setup import *

class TabularModel(object):

    def __init__(self, number_of_states, number_of_actions):
        self._discount = np.zeros((number_of_states, number_of_actions))
        self._reward = np.zeros((number_of_states, number_of_actions))
        self._next_state = np.zeros((number_of_states, number_of_actions))

    def next_state(self, s, a):
        return self._next_state[s][a]

    def reward(self, s, a):
        return self._reward[s][a]

    def discount(self, s, a):
        return self._discount[s][a]

    def transition(self, state, action):
        return (
            self.reward(state, action),
            self.discount(state, action),
            self.next_state(state, action))

    def update(self, state, action, reward, discount, next_state):
        self._discount[state][action] = discount
        self._reward[state][action] = reward
        self._next_state[state][action] = next_state
