from setup import *


class TabularModel(object):

    def __init__(self, number_of_states, number_of_actions):
        self._q = np.zeros((number_of_states, number_of_actions))
        self._m = np.zeros((number_of_states, number_of_actions))
        self._layout = grid._layout


    # a next_state method, taking a state and action and returning the next state in the environment.
    def next_state(self, s, a):
        y, x = s
        if a == 0:  # up
            new_state = (y - 1, x)
        elif a == 1:  # right
            new_state = (y, x + 1)
        elif a == 2:  # down
            new_state = (y + 1, x)
        elif a == 3:  # left
            new_state = (y, x - 1)
        else:
            raise ValueError("Invalid action: {} is not 0, 1, 2, or 3.".format(a))
        #reward, discount, grid.get_obs = grid.step(a)
        return new_state

    # a reward method, taking a state and action and returning the immediate reward associated to execution that action in that state.
    def reward(self, s, a):

        new_y, new_x = s
        if self._layout[new_y, new_x] == -1:  # wall
            reward = -5.
            new_state = (y, x)
        elif self._layout[new_y, new_x] == 0:  # empty cell
            reward = 0.
        else:  # a goal
            reward = self._layout[new_y, new_x]
            discount = 0.
            new_state = self._start_state


        reward, discount, grid.get_obs = grid.step(a)
        return reward

    # a discount method, taking a state and action and returning the discount associated to execution that action in that state.
    def discount(self, s, a):
        reward, discount, grid.get_obs = grid.step(a)
        return discount

    # a transition method, taking a state and an action and returning both the next state and the reward associated to that transition.
    def transition(self, state, action):
        return (
            self.reward(state, action),
            self.discount(state, action),
            self.next_state(state, action))

    # a update method, taking a full transition (state, action, reward, next_state) and updating the model (in its reward, discount and next_state component)
    def update(self, state, action, reward, discount, next_state):
        self._m[state][action] = [reward, next_state]
        self._q[state][action] = reward + discount * next_state - state
        return self._m, self._q


"""
class DynaQ(object):

    def __init__(self, number_of_states, number_of_actions, initial_state, behaviour_policy, num_offline_updates=0, step_size=0.1):
        self._q = np.zeros((number_of_states, number_of_actions))
        self._number_of_states = number_of_states
        self._number_of_actions = number_of_actions
        self._state = initial_state
        self._step_size = step_size
        self._policy = behaviour_policy
        self._updates = num_offline_updates

        self._action = 0 # needs to change that

        self.model = dict()


    @property
    def q_values(self):
        return self._q

    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        return list(state), action, list(next_state), reward


    def step(self, reward, discount, next_state):
        # append most recent observed transition to replay buffer
        s = self._state
        a = self._action
        r = reward
        g = discount
        next_s = next_state



        for _ in range(0, self._updates):
            if tuple(s) not in self.model.keys():
                self.model[tuple(s)] = dict()
            self.model[tuple(s)][a] = [list(next_state), reward]
            state, action, next_state, reward = sample()

        # Update values: Q(s,a) with Q-learning, using transition
        action = self._policy(self._q[s])  # choose action
        td_target = reward + discount * action
        td_error = td_target - self._q[self._state][action]
        self._q[self._state][action] += self._step_size * td_error

        return action

        # Update model: M(s,a)

        # feed the model with previous experience
    #def feed(self, state, action, next_state, reward):
    #    if tuple(state) not in self.model.keys():
    #        self.model[tuple(state)] = dict()
    #    self.model[tuple(state)][action] = [list(next_state), reward]

    # randomly sample n times from previous experience

"""

class DynaQ(object):

    def __init__(self, number_of_states, number_of_actions, initial_state, behaviour_policy, num_offline_updates=0, step_size=0.1):
        self.tab = TabularModel(number_of_states, number_of_actions)
        self._q = np.zeros((number_of_states, number_of_actions))
        self._m = np.zeros((number_of_states, number_of_actions))
        self._layout = grid._layout
        self._start_state = initial_state
        self._number_of_states = number_of_states
        self._number_of_actions = number_of_actions

        self._action = behaviour_policy(self._q)
        self._state = self.tab.next_state(self._start_state, self._action)



    @property
    def q_values(self):
        pass

    def step(self, reward, discount, next_state):



        s = self._state
        a = self._action
        r = reward
        g = discount
        next_s = next_state

        # Update values: Q(s,a) with Q-learning, using transition
        m, q = self.tab.update(s, a, reward, discount, next_state)

        pass


grid = Grid()
agent = DynaQ(
    grid._layout.size, 4, grid.get_obs(),
    random_policy, num_offline_updates=30, step_size=0.1)
run_experiment(grid, agent, int(1e3))
q = agent.q_values.reshape(grid._layout.shape + (4,))
plot_action_values(q)
plt.show()
