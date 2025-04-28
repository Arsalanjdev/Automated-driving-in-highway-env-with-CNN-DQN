import random
from datetime import datetime
from itertools import count
from timeit import default_timer as timer

import numpy as np
import torch
from network import DQN_CNN, Transition, ReplayMemory
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

class DQNAgent:
    def __init__(self, env, config):
        self.env = env
        self.epsilon_start = 0.99
        self.epsilon_end = 0.1
        self.epsilon_decay = 100

        self.batch_size = 128
        self.hidden_size = 128

        self.gamma = 0.99
        self.action_size = env.action_space.n
        self.learning_rate = 0.0001
        self.num_episode = 1000
        self.target_update = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger(self._get_log_path())
        self.memory = ReplayMemory(10000)
        self.step_count = 0
        model = DQN_CNN
        self.policy_network = model(self.action_size, self.hidden_size).to(self.device)
        self.target_network = model(self.action_size, self.hidden_size).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(),
                                       lr=self.learning_rate)
        self.loss_function = F.huber_loss
        self.is_scheduled = False
        self.reward_improvement_threshold = 5.0
        self.early_stopping_patience = 10
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)

    def _get_log_path(self):
        return "./logs/run" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

    def epsilon_greedy_action(self, state):
        self.step_count += 1
        factor = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.step_count / self.epsilon_decay)
        if random.random() > factor:
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.long)

    def compute_q_values(self, state_batch, action_batch):
        return self.policy_network(state_batch).gather(1, action_batch)

    def compute_expected_q_values(self, next_states, rewards):
        mask = torch.tensor([s is not None for s in next_states], device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_states if s is not None])
        v_values = torch.zeros(self.batch_size, device=self.device)
        v_values[mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        return (v_values * self.gamma) + rewards

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).float().to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).float().to(self.device)

        q_values = self.compute_q_values(state_batch, action_batch)
        expected_q_values = self.compute_expected_q_values(batch.next_state, reward_batch)

        loss = self.loss_function(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.is_scheduled:
            self.scheduler.step()
        episode_loss = loss.detach().item()
        return episode_loss

    def train(self):
        cumulated_reward = 0.0
        best_cumulative_reward = -float('inf')
        no_improvement_count = 0
        episode_loss = 0.0

        for episode in range(self.num_episode):
            state, info = self.env.reset()
            state = torch.Tensor(state).unsqueeze(0).to(self.device)

            episode_reward = 0
            episode_start = timer()
            for time_step in count():
                self.env.render()
                action = self.epsilon_greedy_action(state)
                next_state, reward, done, _, info = self.env.step(action.item())
                next_state = torch.Tensor(next_state).unsqueeze(0).to(self.device)
                reward = torch.tensor([reward], device=self.device)

                episode_reward += reward.item()

                self.memory.save_transition(state, action, next_state, reward)
                state = next_state
                step_loss = self.optimize()
                episode_loss += step_loss

                if done:
                    episode_end = timer()
                    episode_length = round(episode_end - episode_start, 6)
                    episode_timesteps = time_step
                    break

            cumulated_reward += episode_reward

            if cumulated_reward > best_cumulative_reward + self.reward_improvement_threshold:
                best_cumulative_reward = cumulated_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= self.early_stopping_patience:
                    print("Stopping early due to no improvement in cumulative reward.")
                    break

            print("Episode:", episode, "|| Timesteps:", time_step, "|| Episode Length:", episode_length, " s", "|| Reward:", episode_reward)

            self.logger.log_scalar("episode_rewards", episode_reward, episode)
            self.logger.log_scalar("episode_lengths", episode_length, episode)
            self.logger.log_scalar("episode_timesteps", episode_timesteps, episode)
            self.logger.log_scalar("cumulated_rewards", cumulated_reward, episode)
            self.logger.log_scalar("loss", episode_loss, episode)

            if episode % self.target_update == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())
