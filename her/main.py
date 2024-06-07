import argparse
import datetime
import numpy as np
import itertools
import torch
import os

from sac import SAC

from envs_robot.register import register_custom_envs
from arguments import get_args
    
    
def get_env_params(env):
    obs = env.reset()[0]
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            'action_min': env.action_space.low[0],  
            }
    params['max_timesteps'] = env._max_episode_steps
    return params


def launch(args):
    # Environment
    if "Fetch" in args.env_name:
        import gymnasium as gym
    elif "Ant" in args.env_name:
        import gym
        
    register_custom_envs()
    env = gym.make(args.env_name, render_mode="rgb_array")
    
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    env_params = get_env_params(env)
    
    sac_trainer = SAC(args, env, env_params)
    sac_trainer.train()
    

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    
    args = get_args()
    launch(args)


                
            
        
