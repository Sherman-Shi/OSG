import torch 
import numpy as np 
import gym 

def evaluate_model(model, env, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = model.predict(torch.tensor(obs).float().unsqueeze(0))
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = action.item()
            elif isinstance(env.action_space, gym.spaces.Box):
                action = action.numpy().flatten()  # Flatten the action
                action_shape = env.action_space.shape
                if action.shape != action_shape:
                    raise ValueError(f"Action generated with shape {action.shape}, but expected {action_shape}.")
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

