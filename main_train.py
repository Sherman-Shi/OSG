import yaml
import wandb
import torch
import gym

import sys
sys.path.append("/home/zhenpeng/桌面/brainstorm/OSG")
from utils.load_model import load_model_from_config
from evaluating.eval import evaluate_model
from data.dataset_sequence import DynamicDataset
from data.dataset_sequence import TargetDataset
from models.diffusion import GaussianDiffusionModel
from models.diffusion import UnconditionalGaussianDiffusionModel
from training.trainer import Trainer




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

    #dynamic
    dynamic_dataset = DynamicDataset(config)    
    dynamic_model = GaussianDiffusionModel(dynamic_dataset, config)
    dynamic_trainer = Trainer(dynamic_dataset, dynamic_model, config, model_type="dynamic_diffusion")

    #target 
    target_dataset = TargetDataset(dynamic_dataset.normalizer, config)
    target_model = UnconditionalGaussianDiffusionModel(target_dataset, config)
    target_traner = Trainer(target_dataset, target_model, config, model_type="target_diffuson")

    n_epochs = int(config['training']['n_total_train_steps'] // config['training']['n_steps_per_epoch'])


    for i in range(n_epochs):
        if config['training']['train_dynamic']:
            dynamic_trainer.train(n_train_steps=config['training']['n_steps_per_epoch'], current_epoch=i)
        if config['training']['train_target']:
            target_traner.train(n_train_steps=config['training']['n_steps_per_epoch'], current_epoch=i)


if __name__ == "__main__":
    main()
