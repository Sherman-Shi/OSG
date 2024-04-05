from models.random import RandomModel

def load_model_from_config(config, action_space):
    model_type = config['model']['type']
    if model_type == 'random':
        return RandomModel(action_space=action_space)
    # Add more model types here as elif statements
    else:
        raise ValueError(f"Unsupported model type: {model_type}")