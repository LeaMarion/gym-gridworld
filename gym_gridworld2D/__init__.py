from gym.envs.registration import register

register(
    id='gridworld2D-v0',
    entry_point='gym_gridworld2D.envs:GridWorld2DEnv',
)

