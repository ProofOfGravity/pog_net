import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, update_target_every=100):
        self.lr = lr
        self.gamma = gamma

        self.policy_net = model
        self.target_net = model

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

        self.update_target_every = update_target_every
        self.step_count = 0

    def train_step(self, replay_buffer, batch_size, gamma):
        self.step_count += 1

        transitions = random.sample(replay_buffer, batch_size)

        states = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        next_states = np.asarray([t[4] for t in transitions])

        states_t = torch.as_tensor(states, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32)

        # Compute targets
        target_q_values = self.target_net(next_states_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + gamma * (1 - dones_t) * max_target_q_values

        q_values = self.policy_net(states_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        self.optimizer.zero_grad()
        loss = self.criterion(action_q_values, targets)
        loss.backward()
        self.optimizer.step()
        self.target_net.load_state_dict(self.policy_net.state_dict())

# for step in itertools.count():

# target_q_values = target_net(new_obses_t)
# max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

# targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

# epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])