import random
import numpy as np
import os

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.episode_stats = {}
        self.item_keys = ['state', 'action', 'reward', 'next_state', 'done', 'episode']

    def push(self, state, action, reward, next_state, done, episode):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, episode)
        
        if episode in self.episode_stats:
            start, _ = self.episode_stats[episode]
            self.episode_stats[episode] = (start, self.position)
        else:
            self.episode_stats[episode] = (self.position, self.position)
        
        if len(self.buffer) == self.capacity:
            overwritten_episode = self.buffer[self.position][-1]
            if overwritten_episode in self.episode_stats:
                start, end = self.episode_stats[overwritten_episode]
                if start == end:
                    del self.episode_stats[overwritten_episode]
                else:
                    self.episode_stats[overwritten_episode] = (start, (self.position + 1) % self.capacity)
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, n_step=1):
        assert len(self.buffer) > 0
        
        if n_step == 1:
            sampled_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        else:
            sampled_indices, n_step = self._sample_n_step(batch_size, n_step)
        
        batch = [self.buffer[i] for i in sampled_indices]
            
        state, action, reward, next_state, done, episode = map(np.stack, zip(*batch))
        
        return state, action, reward, next_state, done, episode, sampled_indices
    
    def _sample_n_step(self, batch_size, n_step):
        assert 'episode' in self.item_keys
        episode_len = {k: (v[1] - v[0]) % self.capacity + 1 for k, v in self.episode_stats.items()}
        episode_len_nstep = {k: v for k, v in episode_len.items() if v >= n_step}
        
        if len(episode_len_nstep) > 0:
            episode_ids = np.array(list(episode_len_nstep.keys()))
            sampled_ids = np.random.choice(episode_ids, size=batch_size, p=None)
            d = self.episode_stats
            lowers = np.array([d[idx][0] for idx in sampled_ids])
            uppers = np.array([d[idx][1] if d[idx][1]>=d[idx][0] else d[idx][1] + self.capacity for idx in sampled_ids]) - (n_step - 1)
            sampled_indices = np.random.randint(lowers, uppers+1) % self.capacity
            batch = [self.buffer[i] for i in sampled_indices]
            for i in range(n_step - 1):
                next_indices = (sampled_indices + i + 1) % self.capacity
                for j, idx in enumerate(next_indices):
                    batch[j][3] = self.buffer[idx][3] # update next_state
                    batch[j][2] += self.buffer[idx][2] # update reward
                    if self.buffer[idx][4]:
                        break # stop accumulating rewards if done
        else:
            raise RuntimeError("No episode is longer than n_step")
        return sampled_indices, n_step
    
    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
