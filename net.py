import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))

        self.std_init = std_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features**0.5
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.sigma_weight.data.fill_(self.std_init / self.in_features**0.5)
        self.sigma_bias.data.fill_(self.std_init / self.out_features**0.5)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.epsilon_weight.copy_(epsilon_out.outer(epsilon_in))
        self.epsilon_bias.copy_(epsilon_out)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.epsilon_weight.copy_(epsilon_out.outer(epsilon_in))
        self.epsilon_bias.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
            bias = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)


class HighwayCNN(nn.Module):
    def __init__(self, device: torch.device, input_channels=4, num_actions=5):
        super().__init__()
        self.device = device
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc1 = NoisyLinear(64 * 32 * 16, 512)
        self.fc2 = NoisyLinear(512, 128)
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 64), nn.ReLU(), NoisyLinear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 64), nn.ReLU(), NoisyLinear(64, num_actions)
        )

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x):
        x: Tensor = x.to(self.device)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        value = self.value_stream(x)
        advantage: Tensor = self.advantage_stream(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
