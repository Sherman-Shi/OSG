import numpy as np

class TargetProcessor:
    def __init__(self, dataset, config):
        """
        Initializes the TrajectoryProcessor with the dataset and configuration.

        Parameters:
        - dataset: The loaded dataset which includes 'observations', 'actions', 'rewards', 'next_observations', and 'timeouts'.
        - config: Configuration object or dict containing 'target_percent' and 'target_len'.
        """
        self.dataset = dataset
        self.config = config

    def _calculate_trajectory_rewards(self):
        """
        Calculates the cumulative reward for each trajectory in the dataset.
        """
        rewards = self.dataset['rewards']
        timeouts = self.dataset['timeouts']
        
        trajectory_rewards = []
        cumulative_reward = 0
        for reward, timeout in zip(rewards, timeouts):
            cumulative_reward += reward
            if timeout:
                trajectory_rewards.append(cumulative_reward)
                cumulative_reward = 0
        # Adding the last trajectory if it does not end with a timeout
        if not timeouts[-1]:
            trajectory_rewards.append(cumulative_reward)
        return trajectory_rewards

    def generate_target_sequences(self):
        """
        Selects the top 10% reward trajectories and organizes the last 50 data points
        as specified for diffusion model training.
        """
        trajectory_rewards = self._calculate_trajectory_rewards()
        reward_threshold = np.percentile(trajectory_rewards, self.config['target']['target_percentile'])  # Top 10%
        
        target_trajectories = []
        current_traj = []
        traj_index = 0
        target_indices = [i for i, r in enumerate(trajectory_rewards) if r >= reward_threshold]
        
        for index, (obs, action, next_obs, reward, timeout) in enumerate(zip(self.dataset['observations'],
                                                                              self.dataset['actions'],
                                                                              self.dataset['next_observations'],
                                                                              self.dataset['rewards'],
                                                                              self.dataset['timeouts'])):
            current_traj.append((obs, action, reward, next_obs))
            if timeout or index == len(self.dataset['rewards']) - 1:
                if traj_index in target_indices:
                    target_trajectories.append(current_traj[-self.config['target']['target_len']:])
                traj_index += 1
                current_traj = []

        target_sequences = {}
        for idx, traj in enumerate(target_trajectories):
            sequence = []
            for obs, action, reward, next_obs in traj:
                sequence.extend([obs, action])  # Assuming reward is not required, adjust if necessary
            target_sequences[f"trajectory_{idx}"] = sequence
        
        return target_sequences
