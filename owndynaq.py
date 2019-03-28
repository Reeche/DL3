from setup import *


class DynaQ(object):

    def __init__(self, number_of_states, number_of_actions, initial_state, behaviour_policy, num_offline_updates=0,
                 step_size=0.1, rand=np.random, discount=0.9):
        self._q = np.zeros((number_of_states, number_of_actions))
        # self._m = np.zeros((number_of_states, number_of_actions))
        self.model = dict()
        self.rand = rand
        self._state = initial_state
        self._number_of_actions = number_of_actions
        self._step_size = step_size
        self._behaviour_policy = behaviour_policy
        self._num_offline_updates = num_offline_updates
        self._layout = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, 0, 0, -1, -1, 0, 0, 10, -1],
            [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
            [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
        self._start_state = (2, 2)
        self._goal_state = (8, 2)
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._discount = discount

    def get_obs(self, s):
        y, x = s
        return y * self._layout.shape[1] + x

    def steps(self, state, action):
        # print("steps state, action", state, action)
        y, x = state

        if action == 0:  # up
            new_state = (y - 1, x)
        elif action == 1:  # right
            new_state = (y, x + 1)
        elif action == 2:  # down
            new_state = (y + 1, x)
        elif action == 3:  # left
            new_state = (y, x - 1)
        else:
            raise ValueError("Invalid action: {} is not 0, 1, 2, or 3.".format(action))

        new_y, new_x = new_state
        if self._layout[new_y, new_x] == -1:  # wall
            reward = -5.
            discount = self._discount
            new_state = (y, x)
        elif self._layout[new_y, new_x] == 0:  # empty cell
            reward = 0.
            discount = self._discount
        else:  # a goal
            reward = self._layout[new_y, new_x]
            discount = 0.
            new_state = self._start_state

        self._state = new_state
        return reward, discount, DynaQ.get_obs(self, self._state)

    @property
    def q_values(self):
        return self._q

    def step(self):

        s = self._state
        a = epsilon_greedy(self._q[DynaQ.get_obs(self, self._state)], 0.1)

        reward, discount, next_state = DynaQ.steps(self, s, a)

        td_target = reward + discount * a
        td_error = td_target - self._q[DynaQ.get_obs(self, s)][a]
        self._q[DynaQ.get_obs(self, s)][a] += self._step_size * td_error

        # update model
        # print("tupel", DynaQ.get_obs(self, s))
        if (DynaQ.get_obs(self, s)) not in self.model.keys():
            self.model[(DynaQ.get_obs(self, s))] = dict()
        self.model[DynaQ.get_obs(self, s)][a] = [(next_state), reward]

        for _ in range(self._num_offline_updates):
            # select random field from _m which is not empty
            state_index = self.rand.choice(range(len(self.model.keys())))
            state = list(self.model)[state_index]
            action_index = self.rand.choice(range(len(self.model[state].keys())))
            action = list(self.model[state])[action_index]
            next_state, reward = self.model[state][action]

            td_target = reward + discount * a
            td_error = td_target - self._q[next_state][a]
            self._q[next_state][a] += self._step_size * td_error

        return action


grid = Grid()
agent = DynaQ(
    grid._layout.size, 4, grid.get_obs(),
    random_policy, num_offline_updates=30, step_size=0.1)
run_experiment(grid, agent, int(1e3))
q = agent.q_values.reshape(grid._layout.shape + (4,))
plot_action_values(q)
plt.show()
