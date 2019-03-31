import matplotlib.pyplot as plt
import numpy as np
#import sonnet as snt
import tensorflow as tf
from collections import namedtuple


# @title Implementation
class Grid(object):

    def __init__(self, discount=0.9):
        # -1: wall
        # 0: empty, episode continues
        # other: number indicates reward, episode will terminate
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

    @property
    def number_of_states(self):
        return self._number_of_states

    def plot_grid(self):
        plt.figure(figsize=(3, 3))
        plt.imshow(self._layout > -1, interpolation="nearest")
        ax = plt.gca()
        ax.grid(0)
        plt.xticks([])
        plt.yticks([])
        plt.title("The grid")
        plt.text(
            self._start_state[0], self._start_state[1],
            r"$\mathbf{S}$", ha='center', va='center')
        plt.text(
            self._goal_state[0], self._goal_state[1],
            r"$\mathbf{G}$", ha='center', va='center')
        h, w = self._layout.shape
        for y in range(h - 1):
            plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], '-k', lw=2)
        for x in range(w - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], '-k', lw=2)

    def get_obs(self):
        y, x = self._state
        return y * self._layout.shape[1] + x

    def int_to_state(self, int_obs):
        x = int_obs % self._layout.shape[1]
        y = int_obs // self._layout.shape[1]
        return y, x

    def step(self, action):
        y, x = self._state

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
        return reward, discount, self.get_obs()


class AltGrid(Grid):

    def __init__(self, discount=0.9):
        # -1: wall
        # 0: empty, episode continues
        # other: number indicates reward, episode will terminate
        self._layout = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
            [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
            [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [-1, 0, 10, 0, 0, 0, 0, 0, 0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
        self._start_state = (2, 2)
        self._goal_state = (2, 7)
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._discount = discount


class FeatureGrid(Grid):

    def get_obs(self):
        return self.state_to_features(self._state)

    def state_to_features(self, state):
        y, x = state
        x /= float(self._layout.shape[1] - 1)
        y /= float(self._layout.shape[0] - 1)
        markers = np.arange(0.1, 1.0, 0.1)
        features = np.array([np.exp(-40 * ((x - m) ** 2 + (y - n) ** 2))
                             for m in markers
                             for n in markers] + [1.])
        return features / np.sum(features ** 2)

    def int_to_features(self, int_state):
        return self.state_to_features(self.int_to_state(int_state))

    @property
    def number_of_features(self):
        return len(self.get_obs())


# @title Show gridworlds

# Plot tabular environments
grid = Grid()
alt_grid = AltGrid()
print("A grid world")
grid.plot_grid()
#plt.show()
print("\nAn alternative grid world")
alt_grid.plot_grid()
#plt.show()

# Plot features of each state for non tabular version of the environment.
print(
    "\nFeatures (visualised as 9x9 heatmaps) for different locations in the grid"
    "\n(Note: includes unreachable states that coincide with walls in this visualisation.)"
)
feat_grid = FeatureGrid()
shape = feat_grid._layout.shape
f, axes = plt.subplots(shape[0], shape[1])
for state_idx, ax in enumerate(axes.flatten()):
    ax.imshow(np.reshape((feat_grid.int_to_features(state_idx)[:-1]), (9, 9)), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
#plt.show()


def run_experiment(env, agent, number_of_steps):
    mean_reward = 0.
    try:
        action = agent.initial_action()
    except AttributeError:
        action = 0
    for i in range(number_of_steps):
        reward, discount, next_state = env.step(action)
        action = agent.step(reward, discount, next_state)
        #action = agent.step()
        mean_reward += (reward - mean_reward) / (i + 1.)

    return mean_reward


map_from_action_to_subplot = lambda a: (2, 6, 8, 4)[a]
map_from_action_to_name = lambda a: ("up", "right", "down", "left")[a]


def plot_rewards(xs, rewards, color):
    mean = np.mean(rewards, axis=0)
    p90 = np.percentile(rewards, 90, axis=0)
    p10 = np.percentile(rewards, 10, axis=0)
    plt.plot(xs, mean, color=color, alpha=0.6)
    plt.fill_between(xs, p90, p10, color=color, alpha=0.3)


def plot_values(values, colormap='pink', vmin=-1, vmax=10):
    plt.imshow(values, interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax)
    plt.yticks([])
    plt.xticks([])
    plt.colorbar(ticks=[vmin, vmax])


def plot_state_value(action_values):
    q = action_values
    fig = plt.figure(figsize=(4, 4))
    vmin = np.min(action_values)
    vmax = np.max(action_values)
    v = 0.9 * np.max(q, axis=-1) + 0.1 * np.mean(q, axis=-1)
    plot_values(v, colormap='summer', vmin=vmin, vmax=vmax)
    plt.title("$v(s)$")


def plot_action_values(action_values):
    q = action_values
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    vmin = np.min(action_values)
    vmax = np.max(action_values)
    dif = vmax - vmin
    for a in [0, 1, 2, 3]:
        plt.subplot(3, 3, map_from_action_to_subplot(a))

        plot_values(q[..., a], vmin=vmin - 0.05 * dif, vmax=vmax + 0.05 * dif)
        action_name = map_from_action_to_name(a)
        plt.title(r"$q(s, \mathrm{" + action_name + r"})$")

    plt.subplot(3, 3, 5)
    v = 0.9 * np.max(q, axis=-1) + 0.1 * np.mean(q, axis=-1)
    plot_values(v, colormap='summer', vmin=vmin, vmax=vmax)
    plt.title("$v(s)$")


def parameter_study(parameter_values, parameter_name,
                    agent_constructor, env_constructor, color, repetitions=10, number_of_steps=int(1e4)):
    mean_rewards = np.zeros((repetitions, len(parameter_values)))
    greedy_rewards = np.zeros((repetitions, len(parameter_values)))
    for rep in range(repetitions):
        for i, p in enumerate(parameter_values):
            env = env_constructor()
            agent = agent_constructor()
            if 'eps' in parameter_name:
                agent.set_epsilon(p)
            elif 'alpha' in parameter_name:
                agent._step_size = p
            else:
                raise NameError("Unknown parameter_name: {}".format(parameter_name))
            mean_rewards[rep, i] = run_experiment(grid, agent, number_of_steps)
            agent.set_epsilon(0.)
            agent._step_size = 0.
            greedy_rewards[rep, i] = run_experiment(grid, agent, number_of_steps // 10)
            del env
            del agent

    plt.subplot(1, 2, 1)
    plot_rewards(parameter_values, mean_rewards, color)
    plt.yticks = ([0, 1], [0, 1])
    plt.ylabel("Average reward over first {} steps".format(number_of_steps), size=12)
    plt.xlabel(parameter_name, size=12)

    plt.subplot(1, 2, 2)
    plot_rewards(parameter_values, greedy_rewards, color)
    plt.yticks = ([0, 1], [0, 1])
    plt.ylabel("Final rewards, with greedy policy".format(number_of_steps), size=12)
    plt.xlabel(parameter_name, size=12)


def random_policy(q):
    return np.random.randint(4)


def epsilon_greedy(q_values, epsilon):
    if epsilon < np.random.random():
        return np.argmax(q_values)
    else:
        return np.random.randint(np.array(q_values).shape[-1])


def plot_greedy_policy(grid, q):
    action_names = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
    greedy_actions = np.argmax(q, axis=2)
    grid.plot_grid()
    plt.hold('on')
    for i in range(9):
        for j in range(10):
            action_name = action_names[greedy_actions[i, j]]
            plt.text(j, i, action_name, ha='center', va='center')


def plot_greedy_policy_v2(grid, pi):
    action_names = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
    greedy_actions = np.argmax(pi, axis=2)
    grid.plot_grid()
    plt.hold('on')

    h, w = grid._layout.shape
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            action_name = action_names[greedy_actions[y - 2, x - 2]]
            plt.text(x, y, action_name, ha='center', va='center')
