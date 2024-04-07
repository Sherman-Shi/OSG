import torch
import numpy as np
from torch import nn, optim
from .model import BaseModel
from .model_utils import cosine_beta_schedule

class GaussianDiffusionModel(BaseModel):
    def __init__(self, config):
        super(GaussianDiffusionModel, self).__init__()
