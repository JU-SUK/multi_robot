import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from sac_clearning import CLearningSAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

from utils import VideoRecorder

from envs_robot.register import register_custom_envs


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="FetchPush_test",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=10000000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--feature_dim', type=int, default=256, metavar='N',)
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_false",
                    help='run on CUDA (default: False)')

parser.add_argument('--gradient_steps_per_epoch', type=int, default=50, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--episodes_per_epoch', type=int, default=8, metavar='N',
                    help='model updates per simulator step (default: 1)')

parser.add_argument('--state_dim', type=int, default=25, metavar='N', help='Fetch Push state dimension (default: 25)')
parser.add_argument('--goal_dim', type=int, default=3, metavar='N', help='Fetch Push goal dimension (default: 3)')
args = parser.parse_args()

# Environment
register_custom_envs()
env = gym.make(args.env_name, render_mode="human")

num_epi =5000
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
if "Fetch" in args.env_name:
    agent = CLearningSAC(env.observation_space['observation'].shape[0]+ args.goal_dim + args.goal_dim , env.action_space, args)
elif "Ant" in args.env_name:
    agent = CLearningSAC(env.observation_space.shape[0]+ args.goal_dim, env.action_space, args)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

agent.load_checkpoint("checkpoints/2024-05-27 16:11:12/sac_checkpoint_{}_{}".format(args.env_name, num_epi), True )

avg_reward = 0.
avg_step = 0.
episodes = 10
skill_index = 0
while True:
    state = env.reset()
    if "Fetch" in args.env_name:
            observations = state[0]["observation"]
            achieved_goal = state[0]["achieved_goal"]
            desired_goal = state[0]["desired_goal"]
            obs = np.concatenate([observations, achieved_goal, desired_goal])
    elif "Ant" in args.env_name:
        observations = state
        desired_goal = state # TODO goal 만들어주는 부분 필요
        obs = np.concatenate([observations, desired_goal])

    episode_reward = 0
    step = 0
    done = False
    while not done:

        action = agent.select_action(obs, evaluate=True)
        next_state, reward, truncated, terminated, _ = env.step(action)
        done = truncated or terminated

        #env.render()
        episode_reward += reward
        step += 1
        if "Fetch" in args.env_name:
            next_observations = next_state["observation"]
            achieved_goal = next_state["achieved_goal"]
            desired_goal = next_state["desired_goal"]
            next_obs = np.concatenate([next_observations, achieved_goal, desired_goal])
        elif "Ant" in args.env_name:
            next_observations = next_state
            desired_goal = next_state
            next_obs = np.concatenate([next_state, desired_goal])
        
        obs = next_obs

    print('Distance :', np.linalg.norm(achieved_goal - desired_goal))
    print('episode_reward :' ,reward)
    print('episode_step :' ,step)
    avg_reward += episode_reward
    avg_step += step