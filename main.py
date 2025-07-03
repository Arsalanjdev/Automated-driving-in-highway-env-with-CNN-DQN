import random

import gymnasium as gym
import highway_env
from langchain_core.utils.mustache import render

from agent import DQNAgent
import torch

from car_agent import CarAgent

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
env = gym.make("highway-v0", config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = CarAgent(4, 5, device)
obs, _ = env.reset()
ep_reward = 0
while True:
    action = agent.act(obs)
    new_obs, reward, done, trunc, info = env.step(action)
    ep_reward += reward
    if done or trunc:
        obs, _ = env.reset()
        print(ep_reward)
        ep_reward = 0
    else:
        obs = new_obs
