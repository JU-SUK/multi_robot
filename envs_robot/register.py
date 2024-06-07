from gymnasium.envs.registration import register

def register_custom_envs():
    register(
        id='FetchPush_test',
        entry_point='envs_robot.fetch.push:MujocoPyFetchPushEnv',
        max_episode_steps=200,
        reward_threshold=0.0,
    )
    
    register(
        id='FetchPickAndPlace_test',
        entry_point='envs_robot.fetch.pick_and_place:MujocoPyFetchPickAndPlaceEnv',
        max_episode_steps=200,
        reward_threshold=0.0,   
    )