import yaml
import wandb
import torch
import gym

import sys
sys.path.append("/home/zhenpeng/桌面/brainstorm/OSG")
from utils.load_model import load_model_from_config
from evaluating.eval import evaluate_model
from data.dataset_sequence import DynamicDataset
from models.diffusion import GaussianDiffusionModel
from training.trainer import Trainer




def load_config(config_path="/home/zhenpeng/桌面/brainstorm/OSG/configs/config.yaml"):
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
    model = GaussianDiffusionModel(dataset, config)
    trainer = Trainer(dataset, model, config)

    n_epochs = int(config['training']['n_total_train_steps'] // config['training']['n_steps_per_epoch'])

    for i in range(n_epochs):
        trainer.train(n_train_steps=config['training']['n_steps_per_epoch'])

if __name__ == "__main__":
    main()
