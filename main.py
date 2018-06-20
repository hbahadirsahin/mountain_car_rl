import random
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from CS545Assignment2.MountainCar import MountainCar
from CS545Assignment2.sarsa import *
from CS545Assignment2.double_q import *
from CS545Assignment2.expected_sarsa import *


NUMBER_OF_RUNS = 0
MAXIMUM_EPISODE_COUNT = 0
ALGORITHM_ID = 0
MAXIMUM_TIME_STEP = 0

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--algorithm',
                        dest='algorithm',
                        type=int,
                        metavar=ALGORITHM_ID,
                        help='Type --algorithm 1 for Sarsa, '
                             '     --algorithm 2 for Expected Sarsa, '
                             '     --algorithm 3 Double-Q Learning'
                             'Default=3',
                        required=False,
                        default=3)
    parser.add_argument('--runs',
                        dest='runs',
                        type=int,
                        metavar=NUMBER_OF_RUNS,
                        help='Number of experiment repetitions. Default=10',
                        required=False,
                        default=10)
    parser.add_argument('--max-episode',
                        dest='max_episode',
                        type=int,
                        metavar=MAXIMUM_EPISODE_COUNT,
                        help='Number of episodes. Default=500',
                        required=False,
                        default=500)
    parser.add_argument('--max-timestep',
                        dest='max_timestep',
                        type=int,
                        metavar=MAXIMUM_TIME_STEP,
                        help='Number of time step before stopping the episode. Default=200',
                        required=False,
                        default=200)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    options = parser.parse_args()

    num_actions = 3  # [move_left, stay, move_right]

    alpha = 0.1
    runs = options.runs
    max_episode = options.max_episode
    algorithm_id = options.algorithm
    max_timestep = options.max_timestep

    Xrange = [-1.2, 0.6]
    Vrange = [-0.07, 0.07]
    start = [random.uniform(-0.6, -0.4), 0]
    goal = [0.5]

    nSteps = [1, 2, 3, 4, 5, 6, 7, 8]

    env = MountainCar(start, goal, Xrange, Vrange)

    if algorithm_id == 1:
        print("Running N-step semi-gradient Sarsa!")
        sarsa_time_steps = np.zeros([len(nSteps), max_episode])
        for run in range(runs):
            print("Starting run", (run+1))
            for n in range(len(nSteps)):
                sarsa_obj = Sarsa(alpha, Xrange, Vrange, max_timestep)
                print("At step:", (n+1))
                for e in range(max_episode):
                    _, time_step, _ = sarsa_obj.run(env, (n+1))
                    sarsa_time_steps[n, e] += time_step
        sarsa_time_steps /= runs
        for n in range(len(nSteps)):
            print("Average number of steps required to finalize an episode for", (n+1), "-Step:",
                  np.average(sarsa_time_steps[n]))
            plt.plot(sarsa_time_steps[n], label='n = '+str(n+1))
        plt.title("Steps per Episode w.r.t. Episode for N-step Semi-gradient Sarsa")
        plt.xlabel("Episode")
        plt.ylabel("Steps per episode")
        plt.legend()
        plt.show()

    elif algorithm_id == 2:
        print("Running N-step semi-gradient Expected Sarsa!")
        for run in range(runs):
            expectedsarsa_time_steps = np.zeros([len(nSteps), max_episode])
            print("Starting run", (run+1))
            expected_sarsa_obj = Expected_Sarsa(alpha, Xrange, Vrange, max_timestep)
            for n in range(len(nSteps)):
                print("At step:", (n+1))
                for e in range(max_episode):
                    _, time_step, _ = expected_sarsa_obj.run(env, (n+1))
                    expectedsarsa_time_steps[n, e] += time_step
        expectedsarsa_time_steps /= runs
        for n in range(len(nSteps)):
            print("Average number of steps required to finalize an episode for", (n+1), "-Step:",
                  np.average(expectedsarsa_time_steps[n]))
            plt.plot(expectedsarsa_time_steps[n], label='n = '+str(n+1))
        plt.title("Steps per Episode w.r.t. Episode for N-step Semi-gradient Expected Sarsa")
        plt.xlabel("Episode")
        plt.ylabel("Steps per episode")
        plt.legend()
        plt.show()

    elif algorithm_id == 3:
        print("Running One-step semi-gradient Double-Q!")
        doubleq_time_steps = np.zeros([1, max_episode])
        for run in range(runs):
            print("Starting run", (run+1))
            doubleq_obj = DoubleQ(alpha, Xrange, Vrange, max_timestep)
            for e in range(max_episode):
                _, time_step, _ = doubleq_obj.run(env, nSteps[0])
                doubleq_time_steps[0, e] += time_step

        doubleq_time_steps /= runs
        print("Average number of steps required to finalize an episode:", np.average(doubleq_time_steps))
        plt.plot(doubleq_time_steps[0], label='n = '+str(nSteps[0]))
        plt.title("Steps per Episode w.r.t. Episode for One-step Semi-gradient Double Q-Learning")
        plt.xlabel("Episode")
        plt.ylabel("Steps per episode")
        plt.legend
        plt.show()

    else:
        print("Invalid algorithm!")


    # for e in range(max_episode):
    #     G, time_step, win_count = expected_sarsa_obj.run(env, n[0])
    #     expected_sarsa_total_win_count += win_count
    #     print("Episode:", e, "Time Step:", time_step, "Win count:", expected_sarsa_total_win_count, "G:", G, "N-Step:", n[0])

    # for e in range(max_episode):
    #     G, time_step, win_count = sarsa_obj.run(env, n[0])
    #     sarsa_total_win_count += win_count
    #     print("Episode:", e, "Time Step:", time_step, "Win count:", sarsa_total_win_count, "G:", G, "N-Step:", n[3])