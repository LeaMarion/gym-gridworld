import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__),
                '../gym_gridworld2D/envs')
                )
from gridworld2D_env import GridWorld2DEnv

# Example: 10 x 10 grid walls
WALLS = [[0, x] for x in [5]] + [[1, x] for x in [0,1,4,5,7,8]] \
                + [[2, x] for x in [1,2,4,7]] + [[3,x] for x in [4,5,6,7,9]] \
                + [[4, x] for x in [1,2]] + [[5,x] for x in [1,6,7,8,9]] \
                + [[6,x] for x in [1,2,3,4,7]] + [[7,x] for x in [2]] \
                + [[8,x] for x in [2,3,4,5,6,7,8,9]]

ENV_PARAMS = {"grid_size": [10, 10],
              "walls_coord": [],
              "max_steps": 400,
              "start_pos":[],
              'reward_pos': [4,4],
              'start': 'all'}

env = GridWorld2DEnv(random_grid=False, valid_grid=False, **ENV_PARAMS)

# Play a few episodes of a random game and render.
for i in range(10):
    observation = env.reset()
    done = False
    env.render()
    print(env.starting_positions)
    while not done:
        (observation, reward, done) = env.step(np.random.choice(range(4)))
        env.render()
        print(observation)
