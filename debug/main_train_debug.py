import yaml
import wandb
import torch
import gym
from utils.load_model import load_model_from_config
from 桌面.brainstorm.OSG.data.dataset_sequence import D4RLDataset  
from data.target import TargetProcessor
from evaluating.eval import evaluate_model 

import sys
sys.path.append("/home/sherman/桌面/rl/osg")

def load_config(config_path="/home/sherman/桌面/rl/osg/configs/config.yaml"):
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

    # Initialize dataset
    dataset = D4RLDataset(config)
    dataset.load_data()

    trajectory_processor = TargetProcessor(dataset.data, config)
    target_sequences = trajectory_processor.generate_target_sequences()

    env_name = config['dataset']['env_name'] 
    env = gym.make(env_name)

    model = load_model_from_config(config, env.action_space)

    # Training and evaluation loop
    eval_freq = config['training']['eval_freq']  
    for epoch in range(config['training']['epochs']):
        
        # Training step
        # model.train_step(...)  # Implement your training logic here

        if epoch % eval_freq == 0 or epoch == config['training']['epochs'] - 1:
            # Perform evaluation
            average_reward = evaluate_model(model, env, num_episodes=10)
            print(f"Epoch {epoch}, Average Reward: {average_reward}")
            
            # Log to wandb
            if config['wandb']['log_to_wandb']:
                wandb.log({"epoch": epoch, "average_reward": average_reward})

    # Save the model after training
    # model.save("path/to/save/model")

if __name__ == "__main__":
    main()
