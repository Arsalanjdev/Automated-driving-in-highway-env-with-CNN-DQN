import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple

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


class ReplayBuffer:
    def __init__(self, device: torch.device, maxlen: int = 512):
        self.queue: deque[Experience] = deque(maxlen=maxlen)
        self.device = device

    def __len__(self):
        return len(self.queue)

    def append(self, exp: Experience):
        state_t = torch.as_tensor(exp.state, dtype=torch.float32, device=self.device)
        next_state_t = torch.as_tensor(
            exp.next_state, dtype=torch.float32, device=self.device
        )
        self.queue.append(
            Experience(
                state_t,
                exp.action,
                exp.reward,
                next_state_t,
                exp.done,
            )
        )

    def batch_to_tensor(
        self, batch: list[Experience]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for exp in batch:
            states.append(exp.state)
            actions.append(
                torch.as_tensor(exp.action, dtype=torch.long, device=self.device)
            )
            rewards.append(
                torch.as_tensor(exp.reward, dtype=torch.float32, device=self.device)
            )
            next_states.append(exp.next_state)
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

    def sample(self, k: int = 128):
        if k >= len(self):
            return self.batch_to_tensor(self.queue)
        else:
            samples = random.sample(self.queue, k=k)
            return self.batch_to_tensor(samples)


class NstepReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        device: torch.device,
        gamma: float = 0.99,
        maxlen: int = 512,
        nstep: int = 5,
    ):
        super().__init__(device=device, maxlen=maxlen)
        self.gamma = gamma
        self.nstep = nstep
        self.temp_buffer: deque[Experience] = deque(maxlen=nstep)

    def append(self, exp: Experience):
        self.temp_buffer.append(exp)
        if len(self.temp_buffer) >= self.nstep or exp.done:
            state = self.temp_buffer[0].state
            action = self.temp_buffer[0].action

            reward = sum(
                (
                    float(exper.reward * (self.gamma**i))
                    for i, exper in enumerate(self.temp_buffer)
                )
            )

            next_state = self.temp_buffer[-1].next_state
            done = self.temp_buffer[-1].done

            replay_experience = Experience(state, action, reward, next_state, done)
            super().append(replay_experience)
            self.temp_buffer.clear()


class CarAgent:
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        device: torch.device,
        epsilon_start: float = 0.99,
        epsilon_stop: float = 0.1,
        epsilon_decay_rate: int = 100000,
        gamma: float = 0.99,
    ):
        self.action_size = action_size
        self.online_net = HighwayCNN(device, obs_size, action_size).to(device)
        self.target_net = HighwayCNN(device, obs_size, action_size).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.epsilon_start = epsilon_start
        self.epsilon_current = epsilon_start
        self.epsilon_stop = epsilon_stop
        self.epsilon_decay_rate = epsilon_decay_rate
        self.gamma = gamma
        self.device = device

    def decay_epsilon(self, steps: int):
        if steps > self.epsilon_decay_rate:
            self.epsilon_current = self.epsilon_stop
        else:
            ratio = steps / self.epsilon_decay_rate
            self.epsilon_current = self.epsilon_start - ratio * (
                self.epsilon_start - self.epsilon_stop
            )

    def act(self, observation):
        if self.epsilon_current > random.random():
            return random.randint(0, self.action_size - 1)
        # tensorize
        obs_t: Tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        obs_t = obs_t.unsqueeze(0)
        self.online_net.eval()
        with torch.no_grad():
            qs: Tensor = self.online_net(obs_t)
            best_q = qs.argmax(dim=1)
        return best_q.item()

    def calculate_loss(self, batch: ReplayBuffer, k: int = 128):
        (
            sampled_state,
            sampled_action,
            sampled_reward,
            sampled_next_state,
            sampled_done,
        ) = batch.sample(k)
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

        return torch.nn.SmoothL1Loss()(current_qs, predicted_qs)
