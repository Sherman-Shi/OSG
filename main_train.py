import yaml
import wandb
from data.dataset import D4RLDataset  
from data.target import TargetProcessor
#from your_model_module import YourModel    

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def initialize_wandb(config):
    # Dynamically import wandb only if logging is enabled
    wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'])
    wandb.config.update(config)

def main():
    # Load configuration
    config = load_config()

    # Initialize wandb
    if config.get('log_to_wandb', False):
        initialize_wandb(config)

    # Initialize dataset
    dataset = D4RLDataset(config)
    dataset.load_data()
    dataset.print_trajectory_info()
    dataset.print_example_datapoints()

    trajectory_processor = TargetProcessor(dataset.data, config)
    target_sequences = trajectory_processor.generate_target_sequences()
    
    # Now you can use target_sequences for your training process
    print(f"Generated {len(target_sequences)} target sequences.")
    # Initialize model
    #model = YourModel(config['model'])


    # Training loop
    for epoch in range(config['training']['epochs']):
        if config.get('log_to_wandb', False):
            wandb.log({"epoch": epoch, "your_metric": 0.0})  # Update with actual metrics

    # Save the model after training
    # model.save("path/to/save/model")

if __name__ == "__main__":
    main()
