import torch
from torch import nn
import os

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Forward pass is not implemented.")

    def train_step(self, *args, **kwargs):
        raise NotImplementedError("Train step is not implemented.")

    def predict(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            return self.forward(x)

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


