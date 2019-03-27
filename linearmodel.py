

class TabularModel(object):

    def __init__(self, number_of_states, number_of_actions):
        pass

    def next_state(self, s, a):
        pass

    def reward(self, s, a):
        pass

    def discount(self, s, a):
        pass

    def transition(self, state, action):
        return (
            self.reward(state, action),
            self.discount(state, action),
            self.next_state(state, action))

    def update(self, state, action, reward, discount, next_state):
        pass