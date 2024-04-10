#
# This dataset code is based on the:
# is conditional generative model all you need for decision making repo
# https://github.com/anuragajay/decision-diffuser
#

import d4rl  
import gym
import torch
import numpy as np
from .dataset_d4rl import load_environment, sequence_dataset, sequence_target_dataset, make_target_indices
from .buffer import ReplayBuffer
from .normalizer import DatasetNormalizer
from collections import namedtuple

Batch = namedtuple('Batch', 'trajectories conditions')

# dataset class 
class DynamicDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.env = load_environment(config["dataset"]["env_name"])
        self.horizon = config["dataset"]["horizon"]
        self.max_episode_len = config["dataset"]["max_episode_len"]
        self.max_n_episodes = config["dataset"]["max_n_episodes"]
        self.termination_penalty = config["dataset"]["termination_penalty"]
        self.normalizer_name = config["dataset"]["normalizer_name"]
        self.use_padding = config["dataset"]["use_padding"]
        self.known_obs_len = config["target"]["known_obs_len"]
        self.target_len = config["target"]["target_len"]
        itr = sequence_dataset(self.env)

        fields = ReplayBuffer(self.max_n_episodes, self.max_episode_len, self.termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, self.normalizer_name, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, self.horizon)
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_episode_len, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_episode_len, -1)
    
    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_episode_len - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices
    
    def get_conditions(self, trajectories):
        '''
            condition on current trajectories for planning
        '''
        known_traj = trajectories[:self.known_obs_len]
        target = trajectories[-self.target_len:]
        return {"known_obs": known_traj, "target": target}
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        trajectories = np.concatenate([actions, observations], axis=-1)
        conditions = self.get_conditions(trajectories)

        batch = Batch(trajectories, conditions)

        return batch



# dataset class 
class TargetDataset(torch.utils.data.Dataset):
    def __init__(self, normalizer, config):
        super().__init__()
        self.env = load_environment(config["dataset"]["env_name"])
        self.max_episode_len = config["dataset"]["max_episode_len"]
        self.max_n_episodes = config["dataset"]["max_n_episodes"]
        self.termination_penalty = config["dataset"]["termination_penalty"]
        self.use_padding = config["dataset"]["use_padding"]
        self.known_obs_len = config["target"]["known_obs_len"]


        ### target dataset specific 
        self.horizon = config["target"]["target_len"] 
        self.target_percentile = config["target"]["target_percentile"]

        itr = sequence_target_dataset(self.env, self.target_percentile)

        fields = ReplayBuffer(self.max_n_episodes, self.max_episode_len, self.termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = normalizer
        self.indices = make_target_indices(fields.path_lengths, self.horizon)
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_episode_len, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_episode_len, -1)
    

    def get_conditions(self, trajectories):
        return {"known_obs": 0, "target": 0}
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        trajectories = np.concatenate([actions, observations], axis=-1)
        conditions = self.get_conditions(trajectories)

        batch = Batch(trajectories, conditions)
        return batch