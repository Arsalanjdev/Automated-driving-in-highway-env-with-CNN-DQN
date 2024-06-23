import random
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F

class DQN_CNN(nn.Module):
    def __init__(self, action_size, hidden_size, input_shape=(1, 4, 100, 100),
                 filter_size=64):
        super(DQN_CNN, self).__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1,stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1,stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.2)

        # self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.drop2 = nn.Dropout(0.2)
        #
        # self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.conv8 = nn.Conv2d(256,512, kernel_size=3)
        # self.pool4 = nn.MaxPool2d(2,2)


        # self.fc1 = nn.Linear(512,hidden_size)
        # self.fc2 = nn.Linear(hidden_size,hidden_size)
        # self.head = nn.Linear(hidden_size,action_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(160000,256)
        self.fc2 = nn.Linear(256,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.head = nn.Linear(hidden_size, action_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)  # Adjust output size as needed
        # self.head = nn.Linear(hidden_size,action_size)

        # def conv_output_size(size, kernel_size=3, stride=1, padding=0, pool_kernel=2, pool_stride=2):
        #     size = ((size - kernel_size + 2 * padding) // stride + 1)
        #     size = (size - pool_kernel) // pool_stride + 1
        #     return size
        #
        # conv_w = conv_output_size(conv_output_size(conv_output_size(input_shape[2])))
        # conv_h = conv_output_size(conv_output_size(conv_output_size(input_shape[3])))
        #
        # conv_output_size = 64 * conv_w * conv_h  # 64 is the output channels of the last conv layer

        # Fully connected layers

        # self.fc1 = nn.Linear(conv_output_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.head = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        x = self.drop1(x)

        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = self.pool2(x)
        # x = self.drop2(x)
        #
        # x = F.relu(self.conv6(x))
        # x = F.relu(self.conv7(x))
        # x = self.pool3(x)
        #
        # x = F.relu(self.conv8(x))


        x = self.flatten(x)  # Use the nn.Flatten layer
        #*(1,512,23,10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc3(x))
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
        self.position += 1
        self.position = self.position % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
