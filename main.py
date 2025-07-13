import random

import gymnasium as gym
import highway_env
import numpy as np
from langchain_core.utils.mustache import render
from matplotlib import pyplot as plt

from agent import DQNAgent
import torch
from torch import Tensor
from car_agent import CarAgent, Experience, NthstepPERBuffer

config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    },
    "policy_frequency": 2,
}


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


env = gym.make("highway-v0", config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = CarAgent(4, 5, device)
buffer = NthstepPERBuffer(512, device)
optimizer = torch.optim.Adam(agent.online_net.parameters(), lr=0.001)
obs, _ = env.reset()
rewards = []
ep_reward = 0
total_steps = 0
ep_loss = 0.0
ep_count = 0
try:
    while True:
        total_steps += 1
        # obs = transpose_observation(obs)
        action = agent.act(obs)
        new_obs, reward, done, trunc, info = env.step(action)
        # new_obs = transpose_observation(new_obs)
        exp = Experience(obs, action, reward, new_obs, done or trunc)
        buffer.append(exp)
        ep_reward += reward

        if total_steps > 512 and total_steps % 4 == 0:
            optimizer.zero_grad()
            agent.online_net.reset_noise()
            agent.online_net.train()
            sampled_batch_list, idx, _ = buffer.sample(128)
            loss: Tensor
            TD_errors: Tensor
            batch = buffer.batch_to_tensor(sampled_batch_list)
            loss, TD_errors = agent.calculate_loss(batch=batch)
            buffer.update(idx, TD_errors)
            loss.backward()
            ep_loss += loss.item()
            optimizer.step()
            torch.cuda.empty_cache()

        if total_steps % 500 == 0:
            agent.target_net.load_state_dict(agent.online_net.state_dict())

        if done or trunc:
            obs, _ = env.reset()
            ep_count += 1
            print(f"Steps: {total_steps}\tReward: {ep_reward}\tLoss: {ep_loss}")
            rewards.append(ep_reward)
            ep_reward = 0
            ep_loss = 0.0
            torch.cuda.empty_cache()
        else:
            obs = new_obs
except KeyboardInterrupt:
    # saving parameters
    torch.save(agent.target_net.state_dict(), "agent_weights.pth")
    print("Parameters saved!")
    smoothed_rewards = moving_average(rewards, window_size=3)
    plt.plot(rewards, label="Raw Rewards")
    plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed Rewards")
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("rewardplot.png")
