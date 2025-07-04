import random

import gymnasium as gym
import highway_env
from langchain_core.utils.mustache import render

from agent import DQNAgent
import torch
from torch import Tensor
from car_agent import CarAgent, ReplayBuffer, Experience, NstepReplayBuffer

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
buffer = NstepReplayBuffer(device, maxlen=512)
optimizer = torch.optim.Adam(agent.online_net.parameters(), lr=0.0001)
obs, _ = env.reset()
ep_reward = 0
total_steps = 0
ep_loss = 0.0
ep_count = 0
while True:
    total_steps += 1
    # obs = transpose_observation(obs)
    action = agent.act(obs)
    agent.decay_epsilon(total_steps)
    new_obs, reward, done, trunc, info = env.step(action)
    # new_obs = transpose_observation(new_obs)
    exp = Experience(obs, action, reward, new_obs, done or trunc)
    buffer.append(exp)
    ep_reward += reward

    if total_steps > 100 and total_steps % 4 == 0:
        optimizer.zero_grad()
        agent.online_net.train()
        loss: Tensor = agent.calculate_loss(batch=buffer)
        loss.backward()
        ep_loss += loss.item()
        optimizer.step()
        torch.cuda.empty_cache()

    if total_steps % 500 == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())

    if done or trunc:
        obs, _ = env.reset()
        ep_count += 1
        print(
            f"Episode: {ep_count}\tReward: {ep_reward}\tLoss: {ep_loss}\tEpsilon: {agent.epsilon_current}"
        )
        ep_reward = 0
        ep_loss = 0.0
        torch.cuda.empty_cache()
    else:
        obs = new_obs
