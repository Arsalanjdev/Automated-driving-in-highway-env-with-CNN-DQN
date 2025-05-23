import gymnasium as gym
import highway_env
from agent import DQNAgent
import torch
torch.cuda.empty_cache()

configs = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (100, 100),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    "policy_frequency": 1,
    "actions": {"type": "DiscreteMetaAction"},
    "vehicles_count": 30,
}

env = gym.make("highway-v0", config=configs)
print(env.reset())

obs, info = env.reset()

agent = DQNAgent(env=env, config=configs)
agent.train()
