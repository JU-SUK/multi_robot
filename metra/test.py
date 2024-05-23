import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from model_metra import Phi, Lambda, Discriminator

from utils_sac import VideoRecorder
from utils_metra import generate_skill, generate_skill_cont

from envs_robot.register import register_custom_envs


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Dual-Ant-v3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=0.01, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=12345, metavar='N',
                    help='random seed (default: 12345)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=100000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=1024, metavar='N',
                    help='hidden size (default: 1024)')

parser.add_argument('--gradient_steps_per_epoch', type=int, default=50, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--episodes_per_epoch', type=int, default=8, metavar='N',
                    help='model updates per simulator step (default: 1)')

parser.add_argument('--start_steps', type=int, default=100, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--skill_dim', type=int, default=16, metavar='N',
                    help='dimension of skill (default: 8)')
parser.add_argument('--cuda', action="store_false",
                    help='run on CUDA (default: True)')
args = parser.parse_args()

# Environment
register_custom_envs()
env_name = "FetchPush_test"
env = gym.make(env_name, render_mode="human")

num_epi = 215000
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Skill dim
skill_dim = 16

# Agent
agent = SAC(env.observation_space['observation'].shape[0] + skill_dim, env.action_space, args)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

agent.load_checkpoint("checkpoints/2024-05-22 16:24:39/sac_checkpoint_{}_{}".format(env_name, num_epi), True )

avg_reward = 0.
avg_step = 0.
episodes = 10
skill_index = 0
while True:
    state = env.reset()
    state = state[0]['observation']
    skill = generate_skill(skill_dim, skill_index)
    state = np.concatenate([state, skill])
    episode_reward = 0
    step = 0
    done = False
    while not done:

        action = agent.select_action(state, evaluate=True)
        next_state, reward, truncated, terminated, _ = env.step(action)
        done = truncated or terminated
        next_state = next_state['observation']
        #env.render()
        episode_reward += reward
        step += 1
        next_state = np.concatenate([next_state[:], skill])
        state = next_state

    print('episode_reward :' ,reward)
    print('episode_step :' ,step)
    print('selected_skill :' ,skill)
    avg_reward += episode_reward
    avg_step += step
    skill_index += 1
    if skill_index == skill_dim:
        break 