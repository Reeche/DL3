from setup import *
from linearmodel import *
from owndynaq import *


class FeatureExperienceQ(ExperienceQ):

    def __init__(
            self, number_of_features, number_of_actions, *args, **kwargs):
        super(FeatureExperienceQ, self).__init__(
            number_of_actions=number_of_actions, *args, **kwargs)
        self._theta = np.zeros((number_of_actions, number_of_features))
        self._linear_model = LinearModel(number_of_features, number_of_actions)

    def q(self, state):
        return np.matmul(self._T, state)

    def step(self, reward, discount, next_state):
        s = self._state
        a = self._action
        r = reward
        g = discount
        next_s = next_state

        # q = np.matmul(self._T[a, :], s)
        q = np.matmul(self._T[a], s)
        q_next = np.matmul(self._T, next_s)

        # self._T[a, :] += self.step_size * (r + g * np.amax(q_next) - q) * s
        self._T[a] += self._step_size * (r + g * np.amax(q_next) - q) * s
        self.theta
        np.matmul(s.T * self.theta)

        self._linear_model.update(s, a, r, g, next_s, self._step_size)

        if len(self._buffer) > (self._num_offline_updates - 1):
            for _ in range(self._num_offline_updates):
                i = np.random.choice(len(self._buffer))
                bs, ba, br, bg, bnext_s = self._buffer[index]

                bq = np.matmul(self._T[ba, :], bnext_s)
                bq_next = np.matmul(self._T, bnext_s)
                self._T[ba] += self._step_size * (br + bg * np.amax(bq_next) - bq) * bnext_s

        self._action = self._behaviour_policy(next_s)
        self._state = next_s
        return self._action


grid = FeatureGrid()

agent = FeatureExperienceQ(
  number_of_features=grid.number_of_features, number_of_actions=4,
  number_of_states=grid._layout.size, initial_state=grid.get_obs(),
  num_offline_updates=0, step_size=0.01, behaviour_policy=random_policy)
run_experiment(grid, agent, int(1e3))
q = np.reshape(
    np.array([agent.q(grid.int_to_features(i)) for i in range(grid.number_of_states)]),
    [grid._layout.shape[0], grid._layout.shape[1], 4])
plot_action_values(q)
plot_greedy_policy(grid, q)
plt.show()