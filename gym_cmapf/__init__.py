from gym.envs.registration import register

register(
    id='cmapf-v0.0.1',
    entry_point='gym_cmapf.envs:CMAPFEnv',
)