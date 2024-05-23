import gymnasium as gym
from envs_robot.register import register_custom_envs
#from gymnasium_robotics.envs.multiagent_mujoco import mujoco_multi
#env = gym.make('FetchReach-v2', render_mode="human")
register_custom_envs()
env = gym.make('FetchPush_test', render_mode="human")
state = env.reset()


while True:
    env.render()
    action = env.action_space.sample()
    state, reward, done, _, _ = env.step(action)

env.close()