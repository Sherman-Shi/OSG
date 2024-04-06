#
# This dataset code is based on the:
# is conditional generative model all you need for decision making repo
# https://github.com/anuragajay/decision-diffuser
#

import d4rl  
import gym
import torch
import numpy as np
from .dataset_d4rl import load_environment, sequence_dataset
from .buffer import ReplayBuffer
from .normalizer import DatasetNormalizer

# dataset class 
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__(config)
        self.env = load_environment(config["dataset"]["env_name"])
        self.horizon = config["dataset"]["horizon"]
        self.max_episode_len = config["dataset"]["max_episode_len"]
        self.max_n_episodes = config["dataset"]["max_n_episodes"]
        self.termination_penalty = config["dataset"]["termination_penalty"]
        itr = sequence_dataset(self.env)

        fields = ReplayBuffer(self.max_n_episodes, self.max_episode_len, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()