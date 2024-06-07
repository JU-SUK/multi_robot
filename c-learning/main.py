import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from sac_clearning import CLearningSAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

from envs_robot.register import register_custom_envs
from utils import VideoRecorder
import datetime

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
env = gym.make(args.env_name, render_mode="rgb_array")
#env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Device
device = torch.device("cuda" if args.cuda else "cpu")

# Video
video_directory = 'video/{}'.format(datetime.datetime.now().strftime("%m-%d %H:%M:%S %p"))
video = VideoRecorder(dir_name=video_directory)

# Agent
if "Fetch" in args.env_name:
    agent = CLearningSAC(env.observation_space['observation'].shape[0]+ args.goal_dim + args.goal_dim , env.action_space, args)
elif "Ant" in args.env_name:
    agent = CLearningSAC(env.observation_space.shape[0]+ args.goal_dim, env.action_space, args)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
episode_idx = 0

for i_epoch in itertools.count(1):
    for i_episode in range(args.episodes_per_epoch):
        episode_reward = 0
        episode_steps = 0
        episode_idx += 1
        
        done = False
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

        while not done:
            #env.render()
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(obs)  # Sample action from policy 
                
            next_state, reward, truncated, terminated, _ = env.step(action)
            done = truncated or terminated
            
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            
            if "Fetch" in args.env_name:
                next_observations = next_state["observation"]
                achieved_goal = next_state["achieved_goal"]
                desired_goal = next_state["desired_goal"]
                next_obs = np.concatenate([next_observations, achieved_goal, desired_goal])
                memory.push(obs, action, reward, next_obs, mask, episode_idx) # Append transition to memory
            elif "Ant" in args.env_name:
                next_observations = next_state
                desired_goal = next_state
                next_obs = np.concatenate([next_observations, desired_goal])
                memory.push(obs, action, reward, next_obs, mask, episode_idx)
                
            obs = next_obs
        
        distance = np.linalg.norm(achieved_goal - desired_goal)
        writer.add_scalar('distance/train', distance, i_episode)
        writer.add_scalar('reward/train', episode_reward, i_episode) 
        print(f"Episode: {episode_idx}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, distance: {distance}")

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.gradient_steps_per_epoch):
                # Update parameters of all the networks
                critic_loss, actor_loss, alpha_loss, alpha_tlogs, critic_loss_label_1, critic_loss_label_2 = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic', critic_loss, updates)
                writer.add_scalar('loss/policy', actor_loss, updates)
                writer.add_scalar('loss/entropy_loss', alpha_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha_tlogs, updates)
                writer.add_scalar('loss/critic_label_1', critic_loss_label_1, updates)
                writer.add_scalar('loss/critic_label_2', critic_loss_label_2, updates)
                updates += 1


    if total_numsteps > args.num_steps:
        break
    
    if episode_idx % 1000 == 0:
        agent.save_checkpoint(args.env_name,f"{episode_idx}")

    if episode_idx % 1000 == 0:
        video.init(enabled=True)
        avg_reward = 0.
        avg_distance = 0.   
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            if "Fetch" in args.env_name:
                observations = state[0]["observation"]
                achieved_goal = state[0]["achieved_goal"]
                desired_goal = state[0]["desired_goal"]
                obs = np.concatenate([observations, achieved_goal, desired_goal])
            elif "Ant" in args.env_name:
                state = state
                desired_goal = state # TODO
                obs = np.concatenate([state, desired_goal])
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(obs, evaluate=True)
                img = env.render()
                video.record(img)

                next_state, reward, truncated, terminated, _ = env.step(action)
                done = truncated or terminated
                
                episode_reward += reward
                
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
            
            avg_reward += episode_reward
            distance = np.linalg.norm(achieved_goal - desired_goal)
        avg_reward /= episodes
        avg_distance /= episodes

        video.save('test_{}.mp4'.format(episode_idx))
        video.init(enabled=False)
        
        writer.add_scalar('avg_distance/test', avg_distance, i_episode)
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()

