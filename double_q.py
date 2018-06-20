import numpy as np
from CS545Assignment2.TileCodingCodeFromSutton import *


class DoubleQ:
    '''
    Note that the code used for tiling is taken from: http://incompleteideas.net/sutton/tiles/tiles3.html
    '''
    def __init__(self, alpha, position_range, velocity_range, max_timestep):
        self.max_num_of_indices = 2048
        self.num_of_tilings = 8
        self.epsilon = 0.1

        self.max_timestep = max_timestep

        self.min_pos = position_range[0]
        self.max_pos = position_range[1]

        self.min_vel = velocity_range[0]
        self.max_vel = velocity_range[1]

        self.alpha = alpha / self.num_of_tilings

        self.iht = IHT(self.max_num_of_indices)

        self.weights1 = -0.001*np.random.rand(self.max_num_of_indices)
        self.weights2 = -0.001*np.random.rand(self.max_num_of_indices)

    def get_action(self, position, velocity, weights=None):
        values = []
        if weights is None:
            greedy_weights = (self.weights1 + self.weights2) / 2
            for action in range(0, 3):
                values.append(self.get_value(position, velocity, action, greedy_weights))
            return np.argmax(values)
        else:
            for action in range(0, 3):
                values.append(self.get_value(position, velocity, action, weights))
            return np.argmax(values)

    def get_active_tiles(self, position, velocity, action):
        return tiles(self.iht,
                     self.num_of_tilings,
                     [self.num_of_tilings * position / (self.max_pos - self.min_pos), self.num_of_tilings * velocity / (self.max_vel - self.min_vel)],
                     [action])

    def get_value(self, position, velocity, action, weights):
        if position == self.max_pos:
            return 0.0
        active_tiles = self.get_active_tiles(position, velocity, action)
        return np.sum(weights[active_tiles])

    def update_weights(self, position, velocity, action, next_position, next_velocity, g):
        new_value = 0
        if np.random.rand() < 0.5:
            active_tiles = self.get_active_tiles(position, velocity, action)
            qhat = np.sum(self.weights1[active_tiles])
            if next_position and next_velocity:
                next_action_weights1 = self.get_action(next_position, next_velocity, self.weights1)
                new_value = self.get_value(next_position, next_velocity, next_action_weights1, self.weights2)
            delta = self.alpha * (g + new_value - qhat)
            for tile in active_tiles:
                self.weights1[tile] += delta,
        else:
            active_tiles = self.get_active_tiles(position, velocity, action)
            qhat = np.sum(self.weights2[active_tiles])
            if next_position and next_velocity:
                next_action_weights2 = self.get_action(next_position, next_velocity, self.weights2)
                new_value = self.get_value(next_position, next_velocity, next_action_weights2, self.weights1)
            delta = self.alpha * (g + new_value - qhat)
            for tile in active_tiles:
                self.weights2[tile] += delta

    def run(self, env, n):
        win_count = 0

        observations = []
        actions = []
        rewards = []

        env.reset()

        # get initial input
        current_observation = env.observe()
        current_action = self.get_action(current_observation[0, 0], current_observation[0, 1])

        observations.append(current_observation)
        actions.append(current_action)
        rewards.append(0.)

        time_step = 0
        T = float('inf')
        while time_step < self.max_timestep:
            time_step += 1

            if time_step < T:
                new_observation, reward, game_over = env.act(current_action)
                observations.append(new_observation)
                rewards.append(reward)

                if reward == 100:
                    win_count += 1

                new_action = self.get_action(new_observation[0, 0], new_observation[0, 1])
                actions.append(new_action)

                if game_over or new_observation[0, 0] == self.max_pos:
                    T = time_step

            updated_time_step = time_step - n
            if updated_time_step >= 0:
                G = 0
                for t in range(updated_time_step + 1, min(T, updated_time_step + n) + 1):
                    G += rewards[t]

                if observations[updated_time_step][0, 0] != self.max_pos:
                    self.update_weights(observations[updated_time_step][0, 0],
                                        observations[updated_time_step][0, 1],
                                        actions[updated_time_step],
                                        observations[updated_time_step+1][0, 0],
                                        observations[updated_time_step+1][0, 1],
                                        G)
            if updated_time_step == T - 1:
                break

            current_action = new_action
        return G, time_step, win_count