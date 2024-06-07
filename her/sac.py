import os
import torch
import datetime
import numpy as np
import itertools
from torch.optim import Adam
import torch.nn.functional as F

from model import DeterministicPolicy, GaussianPolicy, Critic
from utils import hard_update, soft_update

from her import her_sampler
from replay_buffer import replay_buffer
from normalizer import normalizer

from utils import VideoRecorder
from mpi4py import MPI
from torch.utils.tensorboard import SummaryWriter

"""
SAC with HER (Hindsight Experience Replay)
"""

class SAC(object):
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        
        self.lr = args.lr
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n Current device is : ", self.device, "\n" )
        
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(env_params['action']).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
            self.actor = GaussianPolicy(env_params, args.hidden_size).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = DeterministicPolicy(env_params, args.hidden_size).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(env_params, args.hidden_size).to(self.device)
        self.critic_target = Critic(env_params, args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        
        hard_update(self.critic_target, self.critic)
        
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.replay_size, self.her_module.sample_her_transitions)
        
        # normalizer for observation and goal
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        
        # Video
        video_directory = 'video/{}'.format(datetime.datetime.now().strftime("%m-%d %H:%M:%S %p"))
        self.video = VideoRecorder(dir_name=video_directory)
        
        # Tensorboard
        self.writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))
        
    def select_action(self, state, evaluate=False):
        #state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
            
        return action.detach().cpu().numpy()[0]
    
    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return inputs
    
    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        
        # this is for multiprocessing
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]

        
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next
                       }
        
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        
    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g
    
    def update_parameters(self, updates):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        
        # obs, goal normalization & concatenate obs and goal
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        
        # transfer them to tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32).to(self.device)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        rewards_tensor = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device)
        
        # calculate the target Q value function
        with torch.no_grad():
            actions_next, log_pi_next, _ = self.actor.sample(inputs_next_norm_tensor)
            qf1_next_target, qf2_next_target = self.critic_target(inputs_next_norm_tensor, actions_next)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_pi_next
            next_q_value = rewards_tensor +  self.gamma * min_qf_next_target # TODO: check done vs. mask
        
        qf1, qf2 = self.critic(inputs_norm_tensor, actions_tensor)
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        pi, log_pi, _ = self.actor.sample(inputs_norm_tensor)
        qf1_pi, qf2_pi = self.critic(inputs_norm_tensor, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi) 
        
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)
        
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
        
    def train(self):
        total_numsteps = 0
        updates = 0
        episode_idx = 0
        
        for i_epoch in itertools.count(1):
            for i_episode in range(self.args.episodes_per_epoch):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):

                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    episode_steps = 0
                    episode_idx += 1
                    
                    # reset the environment
                    done = False
                    state = self.env.reset()[0]
                    obs = state['observation']
                    ag = state['achieved_goal']
                    g  = state['desired_goal']
                    
                    while not done:
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            if self.args.start_steps > total_numsteps:
                                action = self.env.action_space.sample()
                            else:
                                action = self.select_action(input_tensor)
                        
                        next_state, _, truncated, terminated, info = self.env.step(action)
                        done = truncated or terminated
                        
                        episode_steps += 1
                        total_numsteps += 1
                        
                        mask = 1 if episode_steps == self.env._max_episode_steps else float(not done)
                        
                        obs_new = next_state['observation']
                        ag_new = next_state['achieved_goal']
                        
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                print(f"Episode: {episode_idx}, total_numsteps: {total_numsteps}, episode_steps: {episode_steps}")
                    
                # convert the data to arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                    
                # store the episode
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
            
            if self.buffer.current_size > self.args.batch_size:
                # number of updates per step in environment
                for i in range(self.args.gradient_steps_per_epoch):
                    # update paramters of all the networks
                    critic_1_loss, critic_2_loss, actor_loss, ent_loss, alpha = self.update_parameters(updates)
                    self.writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    self.writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    self.writer.add_scalar('loss/actor', actor_loss, updates)
                    self.writer.add_scalar('loss/ent_loss', ent_loss, updates)
                    self.writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
                        
            if total_numsteps > self.args.num_steps:
                break
            
            if episode_idx % 480 == 0:
                self.save_checkpoint(self.args.env_name, f"{episode_idx}")
            
            if episode_idx % 480 == 0:
                self.video.init(enabled=True)
                episodes = 10
                total_success_rate = []
                for _ in range(episodes):
                    per_success_rate = []
                    state = self.env.reset()[0]
                    obs = state['observation']
                    g = state['desired_goal']
                    done = False
                    while not done:
                        input_tensor = self._preproc_inputs(obs, g)
                        action = self.select_action(input_tensor, evaluate=True)
                        img = self.env.render()
                        self.video.record(img)
                        
                        next_state, _, truncated, terminated, info = self.env.step(action)
                        done = truncated or terminated
                        
                        obs = next_state['observation']
                        g = next_state['desired_goal']
                        per_success_rate.append(info['is_success'])
                    total_success_rate.append(per_success_rate)
                total_success_rate = np.array(total_success_rate)
                local_success_rate = np.mean(total_success_rate[:,-1])
                global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
                success_rate = global_success_rate / MPI.COMM_WORLD.Get_size()
                
                self.video.save('test_{}.mp4'.format(episode_idx))
                self.video.init(enabled=False)
                
                self.writer.add_scalar('success_rate', success_rate, episode_idx)       
                print("-------------------------------------------------------------------------------------------------")
                print(f"Test Episode: {episode_idx}, total_numsteps: {total_numsteps}, episode_steps: {episode_steps}, success_rate: {success_rate}")   
                print("-------------------------------------------------------------------------------------------------")              
                    
    
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists(f'checkpoints/{self.today}/'):
            os.makedirs(f'checkpoints/{self.today}/')
        if ckpt_path is None:
            ckpt_path = f"checkpoints/{self.today}/sac_checkpoint_{env_name}_{suffix}"
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.actor_optim.state_dict()}, ckpt_path)
    
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.actor.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            
            if evaluate:
                self.actor.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.actor.train()
                self.critic.train()
                self.critic_target.train()