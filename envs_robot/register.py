from gymnasium.envs.registration import register

def register_custom_envs():
    register(
        id='FetchPush_test',
        entry_point='envs.fetch.push:MujocoPyFetchPushEnv',
        max_episode_steps=200,
        reward_threshold=0.0,
    )