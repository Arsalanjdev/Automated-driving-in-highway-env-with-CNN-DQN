import random
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F


class DQN_CNN(nn.Module):
    def __init__(self, action_size, hidden_size, input_shape=(1, 4, 100, 100),
                 filter_size=64):
        super(DQN_CNN, self).__init__()
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))  # Output (batch, 256, 5, 5)
        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(256 * 5 * 5, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.head = nn.Linear(hidden_size // 2, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = self.avgpool(x)

        x = self.flatten(x)

        # New improved FC network with careful dropout placement
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Only after first FC

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.head(x)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def save_transition(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
