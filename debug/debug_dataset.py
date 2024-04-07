### this is for debugging the dataset  
import yaml 
import sys
sys.path.append("/home/zhenpeng/桌面/brainstorm/OSG")
from data.dataset_sequence import DynamicDataset


def load_config(config_path="/home/zhenpeng/桌面/brainstorm/OSG/configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    

config = load_config()

dataset = DynamicDataset(config)