import torch
import numpy as np
from torch import nn, optim
from .model_base import BaseModel
from .model_utils import cosine_beta_schedule
from .model_network import TemporalUnet

class GaussianDiffusionModel(BaseModel):
    def __init__(self, dataset, config):
        super(GaussianDiffusionModel, self).__init__()
        self.horizon = dataset.horizon
        self.observation_dim = dataset.observation
        self.action_dim = dataset.action_dim
        self.transition_dim = self.observation_dim + self.action_dim
        self.model = TemporalUnet(self.horizon, self.transition_dim)
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
