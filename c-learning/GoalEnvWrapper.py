import numpy as np
import gym
from envs_robot.fetch import push



class FecthPushEnv(push.MujocoPyFetchPushEnv):
    """
    Wrapper for the Fetchpush environment
    """
    def __init__(self, env):
        super(FecthPushEnv, self).__init__(env)
        
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((50,), -np.inf),
            high=np.full((50,), np.inf),
            dtype=np.float32)
        self.observation_space = self._new_observation_space
        
        
    def step(self, action):
        s, _, _, _ = super(FecthPushEnv, self).step(action)
        done = False
        dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
        r = float(dist < 0.05)
        info = {}
        
        return self.observation(s), r, done, info
    
    
    def reset(self):
        self.observation_space = self._old_observation_space
        s = super(FecthPushEnv, self).reset()
        self.observation_space = self._new_observation_space
        return self.observation(s)
    
    
    def observation(self, observation):
        start_index = 3
        end_index = 6
        goal_pos_1 = observation['achieved_goal']
        goal_pos_2 = observation['observation'][start_index:end_index]
        assert np.all(goal_pos_1 == goal_pos_2)
        s = observation['observation']
        g = np.zeros_like(s)
        g[:start_index] = observation['desired_goal']
        g[start_index:end_index] = observation['desired_goal']
        return np.concatenate([s,g]).astype(np.float32)