import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import wandb

import sys
sys.path.append("/home/zhenpeng/桌面/brainstorm/OSG")
from 桌面.brainstorm.OSG.models.diffusion import UnconditionalDiffusionModel

def normalize_images(images):
    """Normalize image data to [0, 1] range."""
    return (images - images.min()) / (images.max() - images.min())

# Initialize wandb
wandb.init(project='diffusion_model_mnist', entity='sureman0117')

input_dim = 784  # For a flattened 28x28 image
model_dim = 512
n_timesteps = 1000
model = UnconditionalDiffusionModel(input_dim=input_dim, model_dim=model_dim, n_timesteps=n_timesteps)

wandb.config.update({"input_dim": input_dim, "model_dim": model_dim, "n_timesteps": n_timesteps})

# Load dataset
dataset = MNIST(root='/home/zhenpeng/桌面/brainstorm/OSG/MinistData', train=True, transform=ToTensor(), download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

epochs = 100
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

    # Assuming `model` is your trained diffusion model and `generate_samples` generates new images
    batch_size = 16  # Number of samples to generate
    generated_samples = model.generate_samples(batch_size)

    # Normalize the images if not already
    #generated_samples = normalize_images(generated_samples)

    # Reshape if necessary, e.g., for MNIST 1x28x28 to 1x28x28 if it's not already in this shape
    generated_samples = generated_samples.reshape(batch_size, 1, 28, 28)

    # Convert to grid format for easy logging
    # Use torchvision's make_grid if available, or manually create a grid if needed


    wandb.log({"generated_samples": [wandb.Image(generated_samples[i], caption=f"Sample {i}") for i in range(batch_size)]})
wandb.finish()
