import yaml
import wandb
import torch
import gym

from utils.load_model import load_model_from_config
from evaluating.eval import evaluate_model
from data.dataset_sequence import DynamicDataset
from models.diffusion_model import GaussianDiffusionModel

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def initialize_wandb(config):
    wandb.init(project=config['wandb']['project'])
    wandb.config.update(config)

def main():
    # Load configuration
    config = load_config()

    # Initialize wandb
    if config['wandb']['log_to_wandb']:
        initialize_wandb(config)

    dataset = DynamicDataset(config)    
    model = GaussianDiffusionModel(config)
    

if __name__ == "__main__":
    main()
