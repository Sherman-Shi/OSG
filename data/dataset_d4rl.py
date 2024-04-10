import os
import collections
import numpy as np
import gym
import pdb

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    
    wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    dataset = env.get_dataset()
    return dataset

def sequence_dataset(env):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


def sequence_target_dataset(env, target_percentile):
    """
    Returns an iterator through trajectories.
    specific for target dataset 
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    N = dataset['rewards'].shape[0]
    use_timeouts = 'timeouts' in dataset

    episodes = []
    episode_rewards = []
    data_ = collections.defaultdict(list)
    episode_step = 0
    total_reward = 0

    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        
        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])
        total_reward += dataset['rewards'][i]

        if done_bool or final_timestep:
            episode_data = {k: np.array(v) for k, v in data_.items()}
            episodes.append(episode_data)
            episode_rewards.append(total_reward)

            # Reset for the next episode
            data_ = collections.defaultdict(list)
            total_reward = 0
            episode_step = 0
        else:
            episode_step += 1

    # Determine the reward threshold based on the target percentile
    reward_threshold = np.percentile(episode_rewards, target_percentile)

    # Filter episodes
    high_reward_episodes = [episode for episode, reward in zip(episodes, episode_rewards) if reward > reward_threshold]

    return iter(high_reward_episodes)

def make_target_indices(path_lengths, target_len):
    '''
    Creates indices for sampling from dataset where each index maps to a datapoint.
    If the path_length is longer than target_len, it returns the indices for the last target_len segment.
    
    Args:
        path_lengths (list): List of lengths for each path in the dataset.
        target_len (int): The target segment length to sample.
        
    Returns:
        np.ndarray: An array of tuples, each indicating (path_index, start_index, end_index) for sampling.
    '''
    indices = []
    for i, path_length in enumerate(path_lengths):
        if path_length > target_len:
            start = path_length - target_len
            end = path_length
            indices.append((i, start, end))

    if len(indices) == 0:
        raise ValueError("No valid indices found for the given target length and path lengths.")
    indices = np.array(indices)
    return indices

#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
