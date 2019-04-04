from setup import *
from tabularmodel import *


class DynaQ(object):

    def __init__(self, number_of_states, number_of_actions, initial_state, behaviour_policy,
                 num_offline_updates=0, step_size=0.1, discount=0.9):
        self._q = np.zeros((number_of_states, number_of_actions))
        self._state = initial_state
        self._behaviour_policy = behaviour_policy
        self._num_offline_updates = num_offline_updates
        self._step_size = step_size
        self._discount = discount
        self._action = 0
        # self._model = dict()
        self._buffer = []
        self._model = TabularModel(number_of_states, number_of_actions)

    @property
    def q_values(self):
        return self._q

    def step(self, reward, discount, next_state):
        s = self._state
        a = self._action
        r = reward
        g = discount
        next_s = next_state

        self._buffer.append([s, a, r, g, next_s])

        self._q[s][a] += self._step_size * (r + g * np.max(self._q[next_state]) - self._q[s][a])

        # update model
        self._model.update(s, a, r, g, next_s)

        if len(self._buffer) > (self._num_offline_updates - 1):
            for _ in range(self._num_offline_updates):
                i = np.random.choice(len(self._buffer))
                bs, ba, br, bg, bnext_s = self._buffer[i]
                mr, mg, mnext_s = self._model.transition(bs, ba)
                self._q[bs][ba] += self._step_size * (mr + mg * np.amax(self._q[int(mnext_s)]) - self._q[bs][ba])

        # for random policy, the input can be anything. It will always output random number
        self._action = self._behaviour_policy(next_s)
        self._state = next_s

        return self._action


grid = Grid()
agent = DynaQ(
    grid._layout.size, 4, grid.get_obs(),
    random_policy, num_offline_updates=30, step_size=0.1)
run_experiment(grid, agent, int(1e3))
q = agent.q_values.reshape(grid._layout.shape + (4,))
plot_action_values(q)
plt.show()


