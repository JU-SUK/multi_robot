import numpy as np
import logging
import copy
from typing import List, Dict, Tuple, Optional


class ReplayBuffer(object):
    """
    Replay buffer for state transitions.
    
    params:
        :param max_number_of_transitions: Maximum number of transitions to keep
        :param mem_option: Memory option (static, dynamic)
        :param name: Name for indentification (optional)
    returns:
    """
    
    def __init__(
        self,
        max_number_of_transitions: int,
        mem_option: str='static',
        name: str=''
    ) -> None:
        # input args
        self.name = name
        self.capacity = int(max_number_of_transitions)
        self._mem_option = mem_option
        
        # buffer
        self.num_transitions = 0
        self.index = 0
        self.data = None
        self.item_keys = None
        self.episode_stats = {}
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def __getitem__(self, index):
        return {key: d[index] for key, d in zip(self.item_keys, self.data)}
    
    def __setitem__(self, key, value):
        raise TypeError("Insertion forbidden!")
    
    def store(
        self,
        transition: Dict
    ) -> None:
        """
        Store transition

        params:
            : pram transition: Transition to store
        returns:
        """
        # allocating memory (run only once)
        if self.data is None:
            self._preallocate(transition)
        
        # if episode id is provided
        if 'episode' in self.item_keys:
            if self.num_transitions == self.capacity: #consider overwrite
                overwritten_episode_id = self.data[self.item_keys.index('episode')][self.index].tolist()[0]
                s_overwritten, e_overwritten = self.episode_stats[overwritten_episode_id]
                if s_overwritten == e_overwritten:
                    self.episode_stats.pop(overwritten_episode_id)
                else:
                    self.episode_stats[overwritten_episode_id] = ((self.index + 1) % self.capacity, e_overwritten)
            episode_id = int(transition['episode'])
            if episode_id in self.episode_stats:
                s, _ = self.episode_stats[episode_id]
                self.episode_stats[episode_id] = (s, self.index)
            else:
                self.episode_stats[episode_id] = (self.index, self.index)
            
        # add/overwrite transition
        for key, d in zip(self.item_keys, self.data):
            d[self.index] = copy.deepcopy(transition[key]) 
            
        # update num_transitions and index
        self.num_transitions = min(self.num_transitions + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity
            
    def sample(
        self,
        batch_size: int,
        n_step: int=1
    ) -> Dict:
        """
        Randomly sample batch(es) of transition(s) from buffer.
        
        params:
            :param batch_size: Size of batch
            :param n_step: Sample transitions n-step ahead
        returns:
            :return *: Batch(es) of transition(s) of size batch_size
        """
        assert self.num_transitions > 0
        
        def _get_indices(batch_size, n_step):
            if n_step == 1:
                return np.random.choice(self.num_transitions, size=batch_size), 1
            else:
                assert 'episode' in self.item_keys
                episode_len = {k: (v[1] - v[0]) % self.capacity + 1 for k, v in self.episode_stats.items()}
                episode_len_nstep = {k: v for k, v in episode_len.items() if v>=n_step}
                if len(episode_len_nstep) > 0:
                    episode_ids = np.array(list(episode_len_nstep.keys()))
                    sampled_ids = np.random.choice(episode_ids, size=batch_size, p=None)
                    d = self.episode_stats
                    lowers = np.array([d[idx][0] for idx in sampled_ids])
                    uppers = np.array([d[idx][1] if d[idx][1] >= d[idx][0] else d[idx][1] + self.capacity for idx in sampled_ids]) - (n_step-1)
                    return np.random.randint(lowers, uppers+1)%self.capacity, n_step
                else:
                    raise RuntimeError("No episode with length >= n_step found!")
        
        sampled_indices, n_step_ = _get_indices(batch_size, n_step)
        
        ret = {key+'s': d[sampled_indices] for key, d in zip(self.item_keys, self.data)}
        ret['sampled_indices'] = sampled_indices
        for i in range(n_step_-1):
            ret.update({key+'s_%d'%(i+1): d[(sampled_indices+i+1)%self.capacity] for key, d in zip(self.item_keys, self.data)})
        return ret
    
    def clear(self) -> None:
        """
        Reset buffer (does NOT free memory)
        """
        self.num_transitions = 0
        self.index = 0
        self.episode_stats = {}
    
    def _preallocate(
        self,
        transition
    ) -> None:
        """
        Preallocate memory for buffer
        """
        self.item_keys = list(transition.keys())
        transition_np = [np.atleast_1d(np.asarray(transition[key])) for key in self.item_keys]
        # check memory usage
        mem_usage = sum([x.nbytes for x in transition_np]) * self.capacity
        if mem_usage > 10737418240:
            self.logger.warning("Memory usage may exceed 10Gib (name=%s)"%(self.name))
        # preallocate buffer
        if self._mem_option == 'dynamic':
            self.logger.info('Required free memory for replay buffer (name=%s): %.2f MiB'%(self.name, mem_usage/1024/1024))
            self.data = [np.zeros(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
        elif self._mem_option == 'static':
            self.logger.info('Preallocating memory for replay buffer (name=%s): %.2f MiB'%(self.name, mem_usage/1024/1024))
            self.data = [np.ones(dtype=x.dtype, shape=(self.capacity,) + x.shape) for x in transition_np]
        else:
            raise ValueError('Unknown memory option: %s' % self._mem_option)