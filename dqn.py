import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

class DQN(nn.Module):
    """Deep Q-Network."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """DQN Agent implementing experience replay."""
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Store the experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action based on epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, state_size)
        with torch.no_grad():
            act_values = self.model(state)  # Shape: (1, action_size)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        """Train the model based on sampled experiences from memory."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0  # Initialize total loss

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)  # Shape: (1, state_size)
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()
            
            state_tensor = torch.FloatTensor(state)  # Shape: (1, state_size)
            target_f = self.model(state_tensor)  # Shape: (1, action_size)
            target_f[0][action] = target  # Update the Q-value for the selected action

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state_tensor))  # Calculate loss
            total_loss += loss.item()  # Accumulate loss

            loss.backward()
            self.optimizer.step()

        # Print the average loss for this batch
        #print(f"Average Loss: {total_loss / batch_size:.4f}")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
