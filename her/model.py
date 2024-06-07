import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

"""
The input x in both networks should be [o, g], where o is the observation and g is the goal.
"""

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
        

class GaussianPolicy(nn.Module):
    def __init__(self, env_params, hidden_dim):
        super(GaussianPolicy, self).__init__()
        
        self.actor_max = env_params['action_max']
        
        self.mean_linear = nn.Sequential(
            nn.Linear(env_params['obs'] + env_params['goal'], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, env_params['action'])
        )
        
        self.log_std_linear = nn.Sequential(
            nn.Linear(env_params['obs'] + env_params['goal'], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, env_params['action'])
        )
        
        self.apply(weights_init_)
        
        # action rescaling
        self.action_scale = torch.tensor(
            (env_params['action_max'] - env_params['action_min']) / 2., dtype=torch.float32
        )
        self.action_bias = torch.tensor(
            (env_params['action_max'] + env_params['action_min']) / 2., dtype=torch.float32
        )
        
    def forward(self, state):
        mean = self.mean_linear(state)
        log_std = self.log_std_linear(state)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample() # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
        

class DeterministicPolicy(nn.Module):
    def __init__(self, env_params, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        
        self.action_max = env_params['action_max']
        self.action_min = env_params['action_min']
        
        self.mean_linear = nn.Sequential(
            nn.Linear(env_params['obs'] + env_params['goal'], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, env_params['action'])
        )
        
        self.apply(weights_init_)
        
        # action rescaling
        self.action_scale = torch.FloatTensor([float((env_params['action_max'] - env_params['action_min']) / 2.)])
        self.action_bas = torch.FloatTensor([float((env_params['action_max'] + env_params['action_min']) / 2.)])
        
    def forward(self, state):
        x = self.mean_linear(state)
        mean = torch.tanh(x) * self.action_scale + self.action_bias
        
        return mean
    
    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean 
    
class Critic(nn.Module):
    def __init__(self, env_params, hidden_dim):
        super(Critic, self).__init__()
        
        self.action_max = env_params['action_max']
        self.Q1 = nn.Sequential(
            nn.Linear(env_params['obs']+env_params['goal']+env_params['action'], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.Q2 = nn.Sequential(
            nn.Linear(env_params['obs']+env_params['goal']+env_params['action'], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(weights_init_)
    
    def forward(self, obs, action):
        x  = torch.cat([obs, action], dim=1)
        
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        
        return q1, q2
    
    
        
        
        
        