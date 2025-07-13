import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from net import HighwayCNN
import torch.nn as nn


@dataclass
class Experience:
    state: Tensor
    action: int
    reward: float
    next_state: Tensor
    done: bool


class SumTree:
    """
    SumTree data structure that would be used for Prioritized Replay Buffer.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, data, priority: int):
        leaf_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(leaf_index, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, node_index, new_priority):
        delta = new_priority - self.tree[node_index]
        self.tree[node_index] = new_priority
        self._propagate(node_index, delta)

    def _propagate(self, index, delta):
        while index > 0:
            parent = (index - 1) // 2
            self.tree[parent] += delta
            index = parent

    def get_leaf(self, s: float):
        index = self._retrieve(0, s)
        data_index = index - self.capacity + 1
        return index, data_index, self.data[data_index]

    def _retrieve(self, index, s):
        if index >= self.capacity - 1:  # Leaf node
            return index

        left = 2 * index + 1
        right = left + 1

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self):
        return self.tree[0]


class NthstepPERBuffer:
    """
    Prioritized Experience Replay Buffer that gets updated every nth step
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        gamma: float = 0.99,
        nstep: int = 5,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_decay: float = 10000,
    ):
        self.capacity = capacity
        self.gamma = gamma
        self.nstep = nstep
        self.device = device
        self.tree = SumTree(self.capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_decay = beta_decay
        self.step = 1
        self.counter = 1  # counter for nstep
        self.temp_buffer: deque[Experience] = deque(
            maxlen=self.nstep
        )  # temporary buffer for nstep replay buffer

    def _decay_beta(self) -> float:
        return min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * (self.step / self.beta_decay),
        )

    def append(self, exp: Experience):
        self.temp_buffer.append(exp)

        # Check if ready to form n-step experience
        if exp.done or len(self.temp_buffer) == self.nstep:
            state = self.temp_buffer[0].state
            action = self.temp_buffer[0].action
            reward = sum(
                exp.reward * (self.gamma**i) for i, exp in enumerate(self.temp_buffer)
            )
            next_state = self.temp_buffer[-1].next_state
            done = self.temp_buffer[-1].done
            replay_exp = Experience(state, action, reward, next_state, done)

            # Prioritize with max existing priority or 1.0
            max_priority = np.max(self.tree.tree[-self.tree.capacity :])
            max_priority = max_priority if max_priority > 0 else 1.0
            self.tree.add(replay_exp, max_priority)
            self.temp_buffer.clear()

    def batch_to_tensor(
        self, batch: list[Experience]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for exp in batch:
            states.append(
                torch.as_tensor(exp.state, dtype=torch.float32, device=self.device).div(
                    255.0
                )
            )
            actions.append(
                torch.as_tensor(exp.action, dtype=torch.long, device=self.device)
            )
            rewards.append(
                torch.as_tensor(exp.reward, dtype=torch.float32, device=self.device)
            )
            next_states.append(
                torch.as_tensor(
                    exp.next_state, dtype=torch.float32, device=self.device
                ).div(255.0)
            )
            dones.append(
                torch.as_tensor(exp.done, dtype=torch.float32, device=self.device)
            )
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones),
        )

    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], np.ndarray]:
        batch = []
        indices = []
        priorities = []
        beta = self._decay_beta()
        segment = self.tree.total_priority() / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = np.random.uniform(low, high)
            idx, _, exp = self.tree.get_leaf(s)
            indices.append(idx)
            priorities.append(self.tree.tree[idx])
            batch.append(exp)

        priorities_arr = np.array(priorities)
        probs = priorities_arr / self.tree.total_priority()
        is_weights = np.power(self.capacity * probs, -beta)
        is_weights /= is_weights.max()  # Normalize for stability
        self.step += 1
        return batch, indices, is_weights

    def update(self, indices: List[int], errors: Tensor):
        for idx, error in zip(indices, errors):
            priority = (1e-5 + abs(error)) ** self.alpha
            self.tree.update(idx, priority)


class CarAgent:
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        device: torch.device,
        gamma: float = 0.99,
    ):
        self.action_size = action_size
        self.online_net = HighwayCNN(device, obs_size, action_size).to(device)
        self.target_net = HighwayCNN(device, obs_size, action_size).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.gamma = gamma
        self.device = device

    def act(self, observation):
        obs_t: Tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        obs_t = obs_t.unsqueeze(0)
        self.online_net.eval()
        with torch.no_grad():
            qs: Tensor = self.online_net(obs_t)
            best_q = qs.argmax(dim=1)
        return best_q.item()

    def calculate_loss(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], k: int = 128
    ):
        (
            sampled_state,
            sampled_action,
            sampled_reward,
            sampled_next_state,
            sampled_done,
        ) = batch
        sampled_action: Tensor = sampled_action.unsqueeze(1)
        current_qs: Tensor = self.online_net(sampled_state)
        current_qs = current_qs.gather(1, sampled_action).squeeze(1)

        self.target_net.eval()
        with torch.no_grad():
            next_qs_online: Tensor = self.online_net(sampled_next_state)
            next_actions = next_qs_online.argmax(dim=1, keepdim=True)
            # evaluation Double dqn
            evaluated_qs: Tensor = self.target_net(sampled_next_state)
            evaluated_qs = evaluated_qs.gather(1, next_actions).squeeze(1)
        sampled_done = sampled_done.float()

        predicted_qs = sampled_reward + (1.0 - sampled_done) * evaluated_qs * self.gamma
        TD_errors = (current_qs - predicted_qs).detach().abs()

        return (torch.nn.SmoothL1Loss()(current_qs, predicted_qs), TD_errors)
