import numpy as np
from TileCodingCodeFromSutton import *


'''
Note that the code used for tiling is taken from: http://incompleteideas.net/sutton/tiles/tiles3.html
'''
class Expected_Sarsa:

    def __init__(self, alpha, position_range, velocity_range, max_timestep):
        self.max_num_of_indices = 2048
        self.epsilon = 0.1
        self.epsilon_pi = 0.1
        self.num_of_tilings = 8

        self.max_timestep = max_timestep

        self.min_pos = position_range[0]
        self.max_pos = position_range[1]

        self.min_vel = velocity_range[0]
        self.max_vel = velocity_range[1]

        self.alpha = alpha / self.num_of_tilings

        self.iht = IHT(self.max_num_of_indices)

        self.weights = np.zeros(self.max_num_of_indices)

    def get_active_tiles(self, position, velocity, action):
        return tiles(self.iht,
                     self.num_of_tilings,
                     [self.num_of_tilings * position / (self.max_pos - self.min_pos), self.num_of_tilings * velocity / (self.max_vel - self.min_vel)],
                     [action])

    def get_action(self, position, velocity):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 3, size=1)[0]
        else:
            values = []
            for action in range(0, 3):
                values.append(self.get_value(position, velocity, action))
            return np.argmax(values)

    def expected_value(self, position, velocity):
        values = []
        for action in range(0, 3):
            values.append(self.get_value(position, velocity, action))
        return self.epsilon_pi * np.average(values) + (1 - self.epsilon_pi) * np.max(values)

    def get_value(self, position, velocity, action):
        if position == self.max_pos:
            return 0.0
        active_tiles = self.get_active_tiles(position, velocity, action)
        return np.sum(self.weights[active_tiles])

    def update_weights(self, position, velocity, action, q):
        active_tiles = self.get_active_tiles(position, velocity, action)
        qhat = np.sum(self.weights[active_tiles])
        delta = self.alpha * (q - qhat)
        for tile in active_tiles:
            self.weights[tile] += delta

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
                if updated_time_step + n <= T:
                    G += self.expected_value(observations[updated_time_step + n][0, 0],
                                             observations[updated_time_step + n][0, 1])

                if observations[updated_time_step][0, 0] != self.max_pos:
                    self.update_weights(observations[updated_time_step][0, 0],
                                        observations[updated_time_step][0, 1],
                                        actions[updated_time_step], G)
            if updated_time_step == T - 1:
                break

            current_action = new_action
        return G, time_step, win_count
