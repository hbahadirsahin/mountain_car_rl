# Reinforcement Learning - Mountain Car

This repository contains 8 python files:

- MCqlearn.py and MCtest.py are example codes (initially given in the assignment). 
- sarsa.py, expected_sarsa.py and double_q.py have the necessary algorithm classes and functions.
- TileCodingCodeFromSutton.py is the TileCoding code of Sutton's.
- MountainCar.py is the given environment code.
- main.py is the main method to call.

For whom to run this assignment, use following command lines (obviously, one can change the parameters):

- python main.py --algorithm 1 --runs 10 --max-episode 500 --max-timestep 200
- python main.py --algorithm 2 --runs 10 --max-episode 500 --max-timestep 200
- python main.py --algorithm 3 --runs 10 --max-episode 500 --max-timestep 200

Note that main method takes 4 arguments:

- First parameter "--algorithm" defines which algorithm to run and takes integer values 1=Sarsa, 2=Expected Sarsa, 3=Double-Q Learning. As an important note, the default value of this argument is "1". Hence, if you do not provide this argument, the submitted code will always work with Sarsa.
- Second parameter "--runs" defines the number of repetation of the experiment. Its default value is 10. I did not put any control check for this parameter; thus, do not enter 0 or any negative integers. 
- Third parameter "--max-episode" defines the number of maximum episode. Default value is 500.
- Fourth parameter "--max-timestep" defines the number of maximum timesteps to finalize an episode. Default value is 200 (as it is asked in Assignment description).
