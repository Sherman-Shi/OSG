import torch
import numpy as np
from torch import nn, optim
from .model import BaseModel

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

class UnconditionalDiffusionModel(BaseModel):
    def __init__(self, input_dim, model_dim, n_timesteps=1000):
        super(UnconditionalDiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.n_timesteps = n_timesteps
        
        # Using cosine schedule for beta values
        self.betas = cosine_beta_schedule(n_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(torch.from_numpy(self.alphas), dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Model architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, input_dim),
        )

        self.timestep_embedding = nn.Embedding(n_timesteps, model_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, x, t):
        t_emb = self.timestep_embedding(t)
        # Adding timestep embedding to input
        x = x + t_emb
        return self.model(x)

    def sample_q(self, x_start, t):
        """
        Sample from q(x_t | x_0) at timestep t
        """
        noise = torch.randn_like(x_start)
        return (
            self.sqrt_alphas_cumprod[t].to(x_start.device) * x_start +
            self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device) * noise
        )

    def train_step(self, x_start):
        """
        Performs a training step.
        """
        self.train()
        t = torch.randint(0, self.n_timesteps, (x_start.size(0),), device=x_start.device)
        x_noised = self.sample_q(x_start, t)
        x_recon = self.forward(x_noised, t)
        
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(x_recon, x_start)  # Reconstruct x_start from x_noised
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def generate_samples(self, batch_size):
        """
        Generate samples from the model.
        """
        x = torch.randn(batch_size, self.input_dim, device=self.model[0].weight.device)
        for t in reversed(range(self.n_timesteps)):
            x = self.forward(x, torch.full((batch_size,), t, dtype=torch.long, device=x.device))
        return x