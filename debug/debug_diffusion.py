import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import wandb

import sys
sys.path.append("/home/sherman/桌面/rl/osg")
from models.diffusion_model import UnconditionalDiffusionModel

# Initialize wandb
wandb.init(project='diffusion_model_mnist', entity='sureman0117')

input_dim = 784  # For a flattened 28x28 image
model_dim = 512
n_timesteps = 1000
model = UnconditionalDiffusionModel(input_dim=input_dim, model_dim=model_dim, n_timesteps=n_timesteps)

wandb.config.update({"input_dim": input_dim, "model_dim": model_dim, "n_timesteps": n_timesteps})

# Load dataset
dataset = MNIST(root='/home/sherman/桌面/rl/osg/MinistData', train=True, transform=ToTensor(), download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

epochs = 5
for epoch in range(epochs):
    epoch_loss = 0
    for batch_idx, (x_start, _) in enumerate(data_loader):
        x_start = x_start.view(x_start.size(0), -1)  # Flatten the images
        loss = model.train_step(x_start)
        epoch_loss += loss
        wandb.log({"batch_loss": loss})

    epoch_loss /= len(data_loader.dataset)
    wandb.log({"epoch_loss": epoch_loss})
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

# Generate and log samples
with torch.no_grad():
    samples = model.generate_samples(batch_size=16).view(-1, 1, 28, 28)  # Reshape to image format
    samples_grid = make_grid(samples, nrow=4)
    # Log generated samples to wandb
    wandb.log({"generated_samples": [wandb.Image(samples_grid, caption="Generated Samples")]})

wandb.finish()
