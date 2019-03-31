from setup import *


class DynaQ(object):

    def __init__(self, number_of_states, number_of_actions, initial_state, behaviour_policy, num_offline_updates=0,
                 step_size=0.1, discount=0.9):
        self._q = np.zeros((number_of_states, number_of_actions))
        self._state = initial_state
        self._behaviour_policy = behaviour_policy
        self._num_offline_updates = num_offline_updates
        self._step_size = step_size
        self._discount = discount
        self._action = 0
        self._model = dict()

    @property
    def q_values(self):
        return self._q

    def step(self, reward, discount, next_state):
        s = self._state
        a = self._action
        r = reward
        g = discount
        next_s = next_state

        # for random policy, the input can be anything. It will always output random number
        self._action = self._behaviour_policy(next_s)

        self._q[s][a] += self._step_size * (r + g * np.max(self._q[next_state]) - self._q[s][a])


        # update model
        if s not in self._model.keys():
            self._model[s] = dict()
        # here a or a_random?
        self._model[s][a] = [r, g, next_s]
        #print(self._model[s][a])

        # replay
        for _ in range(self._num_offline_updates):
            ms = np.random.choice(list(self._model.keys()))
            ma = np.random.choice(tuple(self._model[ms]))
            #print(self._model[ms][ma])

            mr, mg, mnext_s = self._model[ms][ma]
            #print(mr, mg, mnext_s)

            self._q[ms][ma] += self._step_size * (mr + mg * np.max(self._q[mnext_s]) - self._q[ms][ma])

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
