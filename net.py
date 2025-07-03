import torch
import torch.nn as nn
from torch import Tensor


class HighwayCNN(nn.Module):
    def __init__(self, device: torch.device, input_channels=4, num_actions=5):
        super().__init__()
        self.device = device
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x: Tensor = x.to(self.device)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
