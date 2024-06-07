import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
    
    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        # this is for multiprocessing
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        
        # Select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p) # her 샘플링에 포함될 인덱스를 나타냄
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples) # 현재 시점에서 에피소드의 끝까지 남은 타임스텝 수 계산해서 각 샘플에 대한 미래 시점의 오프셋을 계산
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes] # her sampling에 사용될 미래 시점을 계산
        
        # Replace desired goal with acheived goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        
        # Re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        
        
        return transitions