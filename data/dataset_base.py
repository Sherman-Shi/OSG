import d4rl  # Make sure to install d4rl with pip
import gym
import numpy as np

class BaseDataset:
    def __init__(self, config):
        self.config = config
        self.data = None

    def load_data(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def preprocess_data(self):
        raise NotImplementedError("This method should be overridden by subclasses")

class D4RLDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.env = None

    def load_data(self):
        self.env = gym.make(self.config['dataset']['env_name'])

        self.data = self.env.get_dataset()
        self.preprocess_data()

    def preprocess_data(self):
        pass


    def print_trajectory_info(self):
        timeouts = self.data['timeouts']
        rewards = self.data['rewards']
        num_trajectories = sum(timeouts)
        total_samples = len(timeouts)
        average_length = total_samples / num_trajectories if num_trajectories > 0 else 0
        total_rewards = np.sum(rewards)
        average_reward = total_rewards / total_samples if total_samples > 0 else 0
        
        print(f"Number of trajectories in the dataset: {num_trajectories}")
        print(f"Total number of samples (data points): {total_samples}")
        print(f"Average trajectory length: {average_length:.2f}")
        print(f"Total rewards accumulated across all trajectories: {total_rewards}")
        print(f"Average reward per sample: {average_reward:.2f}")

        # Optionally, you can add more detailed analysis, like reward distribution
        # For instance, print the minimum and maximum reward observed
        print(f"Minimum reward in the dataset: {np.min(rewards)}")
        print(f"Maximum reward in the dataset: {np.max(rewards)}")


    def print_example_datapoints(self, num_examples=5):
        """
        Prints example data points from the dataset.
        
        Parameters:
            num_examples (int): Number of examples to print.
        """
        print(f"Example datapoints (showing up to {num_examples}):")
        
        states = self.data['observations']
        actions = self.data['actions']
        rewards = self.data['rewards']
        next_states = self.data['next_observations']
        
        for i in range(min(num_examples, len(states))):
            print(f"\nDatapoint {i+1}:")
            print(f"State: {states[i]}")
            print(f"Action: {actions[i]}")
            print(f"Reward: {rewards[i]}")
            print(f"Next State: {next_states[i]}")