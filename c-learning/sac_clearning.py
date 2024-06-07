import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model_clearning import GaussianPolicy, Critic, DeterministicPolicy
import numpy as np
import datetime

class CLearningSAC(object):
    def __init__(self, num_inputs, action_space, args, w_clip=20, lambda_coef=1.0, mu_coef=1.0, use_reward=False, actor_loss_type='logC'):
        self.state_dim = args.state_dim
        self.goal_dim = args.goal_dim
        
        
        self.w_clip = w_clip
        self.use_reward = use_reward
        self.actor_loss_type = actor_loss_type
        
        self.relabel_next_ratio = lambda_coef / 2.0
        self.relabel_future_ratio = (1 - lambda_coef) / 2.0
        self.relabel_random_ratio = mu_coef / 2.0
        self.relabel_original_ratio = (1 - mu_coef) / 2.0
        
    
        self.env_name = args.env_name    
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.lr = args.lr
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        
        self.device = torch.device("cuda" if args.cuda else "cpu")
        
        self.today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n Current device is : ", self.device, "\n" )
        
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
            
            self.actor = GaussianPolicy(num_inputs, action_space.shape[0], args.feature_dim, args.hidden_size, action_space).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = DeterministicPolicy(num_inputs, action_space.shape[0], args.feature_dim, args.hidden_size, action_space).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        
        self.critic = Critic(num_inputs, action_space.shape[0], args.feature_dim, args.hidden_size).to(self.device)
        self.critic_target = Critic(num_inputs, action_space.shape[0], args.feature_dim, args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        hard_update(self.critic_target, self.critic)
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]
        
        
    def update_critic(self, state_batch, action_batch, next_state_batch, batch_size, reward=None):
        
        gamma = self.gamma
        log_gamma = np.log(gamma)
        lambda_coef = 2 * self.relabel_next_ratio
        log_lambda = np.log(lambda_coef)
        w_clip = 1/(1 - gamma) if self.w_clip is None else self.w_clip
        
        batch_next = int(batch_size*self.relabel_next_ratio)
        batch_future = int(batch_size*self.relabel_future_ratio)
        batch_random_original = batch_size - batch_next - batch_future
        
        # calculate the w
        with torch.no_grad():
            next_action_batch, next_action_log_pi, _ = self.actor.sample(next_state_batch)
            next_target_logit1, next_target_logit2 = self.critic_target(next_state_batch, next_action_batch)
            target_logit = torch.min(next_target_logit1, next_target_logit2) - self.alpha * next_action_log_pi # TODO: SAC라서 뒤의 term??
            if w_clip > 0: # w = exp(logit)
                target_logit = torch.clamp(target_logit, None, np.log(w_clip))
            w = torch.exp(target_logit)
            #w = torch.exp(target_logit) / (1 - torch.exp(target_logit)) # TODO: 뭐가 맞는거지???
            td_target = torch.sigmoid(target_logit + log_gamma + log_lambda) # td_target = lambda * gamma * w/(1 + lambda * gamma * w)
            
            label1 = torch.ones([batch_next + batch_future, 1]).to(self.device)
            label2 = td_target[-batch_random_original:] if reward is None else torch.where(reward[-batch_random_original:] == 1.0, 1.0, td_target[-batch_random_original:]).to(self.device)
            label = torch.concat([label1, label2], axis=0)
            
            weight1 = (1-gamma)*torch.ones([batch_next, 1]).to(self.device)
            weight2 = torch.ones([batch_future, 1]).to(self.device)
            weight3 = (1 + lambda_coef*gamma*w)[-batch_random_original:] if reward is None else torch.where(reward[-batch_random_original:] == 1.0, 1-gamma, (1+lambda_coef*gamma*w)[-batch_random_original:]).to(self.device)  
            weight = torch.concat([
                weight1,
                weight2,
                weight3
            ], axis=0)

        logit1, logit2 = self.critic(state_batch, action_batch)
        loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        
        clearning_loss = 0.5*(loss(logit1, label) + loss(logit2, label))
        loss = torch.nn.BCEWithLogitsLoss()
        loss_min = torch.nn.BCELoss()
        n = batch_next + batch_future
        clearning_loss_label_1 = 0.5*(loss(logit1[:n], label[:n]) + loss(logit2[:n], label[:n])) - 2*loss_min(label[:n], label[:n])
        n = batch_random_original
        clearning_loss_label_2 = 0.5*(loss(logit1[-n:], label[-n:]) + loss(logit2[-n:], label[-n:])) - 2*loss_min(label[-n:], label[-n:])
        
        self.critic_optim.zero_grad()
        clearning_loss.backward()
        self.critic_optim.step()
        
        return clearning_loss.item(), clearning_loss_label_1.item(), clearning_loss_label_2.item()
    
    def update_actor(self, state_batch):
        
        action, action_log_pi, _ = self.actor.sample(state_batch)
        
        logit1, logit2 = self.critic(state_batch, action)
        logit = torch.min(logit1, logit2)
        
        if self.actor_loss_type == 'logC':
            m = torch.nn.LogSigmoid()
        elif self.actor_loss_type == 'C':
            m = torch.nn.Sigmoid()
        elif self.actor_loss_type == 'logitC':
            m = torch.nn.Identity()
        
        actor_loss = -m(logit) + self.alpha * action_log_pi
        actor_loss = actor_loss.mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        return actor_loss.item()      
    
    def transition_tuple(self, memory, batch_size):
        """
        buffer sample + relabeling(next & random)
        2가지 역할 수행
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, episode_batch, sampled_idx = memory.sample(batch_size=batch_size)
        
        # batch size
        batch_next = int(batch_size*self.relabel_next_ratio)
        batch_future = int(batch_size*self.relabel_future_ratio)
        batch_random = int(batch_size*self.relabel_random_ratio)
        batch_original = batch_size - batch_next - batch_future - batch_random
        batch_sizes = [batch_next, batch_future, batch_random, batch_original]
        
        # split
        obs_batches = np.split(state_batch, np.cumsum(batch_sizes), axis=0)
        reward_batches = np.split(reward_batch, np.cumsum(batch_sizes), axis=0)
        next_obs_batches = np.split(next_state_batch, np.cumsum(batch_sizes), axis=0)
        if batch_future > 0:
            sampled_episode_ends = np.array([memory.episode_stats[ids][1] for ids in sampled_idx])
            sampled_idx_batches = np.split(sampled_idx, np.cumsum(batch_sizes), axis=0)
            sampled_epiend_batches = np.split(sampled_episode_ends, np.cumsum(batch_sizes), axis=0)
        
        # relabel
        obs_batches[0], reward_batches[0], next_obs_batches[0] = self.relabel_fn(obs_batches[0], reward_batches[0], next_obs_batches[0], goal_cand=next_obs_batches[0], relabel_goal='next')
        obs_batches[2], reward_batches[2], next_obs_batches[2] = self.relabel_fn(obs_batches[2], reward_batches[2], next_obs_batches[2], goal_cand=state_batch, relabel_goal='random')
        obs_batches[3], reward_batches[3], next_obs_batches[3] = self.relabel_fn(obs_batches[3], reward_batches[3], next_obs_batches[3], goal_cand=None)
        # relabel future
        if batch_future > 0:
            horizon = (sampled_epiend_batches[1] - sampled_idx_batches[1])%memory.capacity
            future_idx_batch = np.random.geometric(p=self.gamma, size=horizon.shape)
            future_idx_batch (sampled_idx_batches[1] + future_idx_batch) % memory.capacity
            future_obs_batch = np.array([memory.buffer[i][3] for i in future_idx_batch]) # future next_state # since we want to sample s_t+
            obs_batches[1], reward_batches[1], next_obs_batches[1] = \
                self.relabel_fn(obs_batches[1], reward_batches[1], next_obs_batches[1], goal_cand=future_obs_batch, relabel_goal='future')
        
        # concatenate
        obs = np.concatenate(obs_batches, axis=0)
        reward = np.concatenate(reward_batches, axis=0)
        next_obs = np.concatenate(next_obs_batches, axis=0)
        
        memory_relabel = [obs, action_batch, reward, next_obs, mask_batch]
        return memory_relabel
    
    def relabel_fn(self, obs, reward, next_obs, goal_cand, relabel_goal=None):
        if relabel_goal is None:
            obs_, reward_, next_obs_ = obs, reward, next_obs
        elif relabel_goal == 'next':
            next_goal = self.get_achieved_goal(goal_cand)
            obs_ = self.get_obs(obs, desired_goal=next_goal)
            next_obs_ = self.get_obs(next_obs, desired_goal=next_goal)
            reward_ = np.ones_like(reward)
        elif relabel_goal == 'random':
            idx = np.random.randint(goal_cand.shape[0], size=obs.shape[0])
            random_goal = self.get_achieved_goal(goal_cand[idx])
            obs_ = self.get_obs(obs, desired_goal=random_goal)
            next_obs_ = self.get_obs(next_obs, desired_goal=random_goal)
            reward_ = self.compute_reward(self.get_achieved_goal(next_obs_), self.get_desired_goal(obs_))
        elif relabel_goal == 'future':
            future_goal = self.get_achieved_goal(goal_cand)
            obs_ = self.get_obs(obs, desired_goal=future_goal)
            next_obs_ = self.get_obs(next_obs, desired_goal=future_goal)
            reward_ = self.compute_reward(self.get_achieved_goal(next_obs_), self.get_desired_goal(obs_))
        else:
            raise ValueError(f"Unknown relabel_goal: {relabel_goal}")
        return obs_.reshape(obs.shape), reward_.reshape(reward.shape), next_obs_.reshape(next_obs.shape)
    
    def get_state(self, obs):
        obs = np.atleast_2d(obs)
        return np.squeeze(obs[:, :self.state_dim]).astype("float32")
    
    def get_achieved_goal(self, obs):
        obs = np.atleast_2d(obs)
        if 'Fetch' in self.env_name:
            return np.squeeze(obs[:, self.state_dim:-self.goal_dim]).astype("float32")
        elif 'Ant' in self.env_name:
            return self.get_state(obs).astype("float32")
    
    def get_desired_goal(self, obs):
        obs = np.atleast_2d(obs)
        return np.squeeze(obs[:, -self.goal_dim:]).astype("float32")
    
    def get_obs(self, obs=None, state=None, achieved_goal=None, desired_goal=None):
        if obs is None:
            assert state is not None and achieved_goal is not None and desired_goal is not None
            state = np.atleast_2d(state)
            achieved_goal = np.atleast_2d(achieved_goal)    
            desired_goal = np.atleast_2d(desired_goal)
        else:
            state = np.atleast_2d(self.get_state(obs) if state is None else state)
            achieved_goal = np.atleast_2d(self.get_achieved_goal(obs) if achieved_goal is None else achieved_goal)
            desired_goal = np.atleast_2d(self.get_desired_goal(obs) if desired_goal is None else desired_goal)
        
        if 'Fetch' in self.env_name:
            return np.squeeze(np.concatenate([state, achieved_goal, desired_goal], axis=1)).astype("float32")
        elif 'Ant' in self.env_name:
            return np.squeeze(np.concatenate([state, desired_goal], axis=1)).astype("float32")
        else:
            raise NotImplementedError
          
    
    def update_alpha(self, state_batch):
        action, action_log_pi, _ = self.actor.sample(state_batch)
        
        self.alpha_optim.zero_grad()
        alpha_loss = -(self.log_alpha * (action_log_pi + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optim.step()
        
        return alpha_loss.item()
    
    
    def update_parameters(self, memory, batch_size, updates):
        # sample a batch from memory
        #state_batch, action_batch, reward_batch, next_state_batch, mask_batch, episode, sampled_indices  = memory.sample(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.transition_tuple(memory, batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        # update critic
        if not self.use_reward:
            reward = None
            critic_loss, critic_loss_label_1, critic_loss_label_2 = self.update_critic(state_batch, action_batch, next_state_batch, batch_size, reward=reward)
        
        # update actor
        actor_loss = self.update_actor(state_batch)
        
        # update alpha
        if self.automatic_entropy_tuning:
            alpha_loss = self.update_alpha(state_batch)
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For tensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For tensorboardX logs
            
        # update critic target
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        return critic_loss, actor_loss, alpha_loss, alpha_tlogs, critic_loss_label_1, critic_loss_label_2
        
    # Save model parameters
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
    
    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
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
        
            
    @staticmethod
    def compute_reward(achieved_goal, desired_goal):
        return np.array(achieved_goal == desired_goal).astype(np.float32)
        
        
        