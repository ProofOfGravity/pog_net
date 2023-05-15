import math
from math import floor

import pygame
import sys
import numpy as np
import random
import pandas as pd
from collections import namedtuple, deque
import os

from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        state, action, next_state, reward = args
        state = torch.from_numpy(state).float().unsqueeze(0)  # Convert to float and add batch dimension
        action = torch.tensor([action])  # Assume action is a simple integer.

        next_state = torch.from_numpy(next_state).float().unsqueeze(
            0) if next_state is not None else None  # Convert to float
        reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        return self.layer3(x)


loss_fn = torch.nn.SmoothL1Loss()

GAMMA = 0.999
BATCH_SIZE = 128
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
LR = 5e-4


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).unsqueeze(-1)
    reward_batch = torch.cat(batch.reward)




    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


pygame.init()

# Constants
GRID_SIZE = 20
CELL_SIZE = 30
WIDTH = HEIGHT = CELL_SIZE * GRID_SIZE
BACKGROUND_COLOR = (0, 0, 0)
GRID_COLOR = (128, 128, 128)
X_COLOR = (255, 0, 0)
O_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
FONT_SIZE = 30

# Global variables
x_x_pos, x_y_pos = 0, 0
o_x_pos, o_y_pos = GRID_SIZE - 1, GRID_SIZE - 1
x_max, y_max = GRID_SIZE - 1, GRID_SIZE - 1
turn_counter = 0
x_reward, o_reward = 0, 0
match_over = False

actions = [0, 1, 2, 3, 4]

# Initialize PyGame components
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('X and O Game')
clock = pygame.time.Clock()
font = pygame.font.Font(None, FONT_SIZE)


# Helper functions
def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))


def draw_x(x, y):
    text = font.render('X', True, X_COLOR, BACKGROUND_COLOR)
    text_rect = text.get_rect()
    text_rect.center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
    screen.blit(text, text_rect)


def draw_o(x, y):
    text = font.render('O', True, O_COLOR, BACKGROUND_COLOR)
    text_rect = text.get_rect()
    text_rect.center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
    screen.blit(text, text_rect)


def calculate_state():
    global x_x_pos, x_y_pos, o_x_pos, o_y_pos, GRID_SIZE
    x_x = x_x_pos
    x_y = x_y_pos
    o_x = o_x_pos
    o_y = o_y_pos


    state = np.zeros((GRID_SIZE * GRID_SIZE), dtype=int)

    state[x_x + (x_y * GRID_SIZE)] = 1
    state[o_x + (o_y * GRID_SIZE)] = 2

    return state


def draw_turn_counter():
    text = font.render(f'Turn: {turn_counter}', True, TEXT_COLOR, BACKGROUND_COLOR)
    text_rect = text.get_rect()
    text_rect.topright = (WIDTH - 10, 10)
    screen.blit(text, text_rect)


def draw_x_reward():
    text = font.render(f'X Reward: {x_reward}', True, TEXT_COLOR, BACKGROUND_COLOR)
    text_rect = text.get_rect()
    text_rect.bottomleft = (10, HEIGHT - 10)
    screen.blit(text, text_rect)


def draw_o_reward():
    text = font.render(f'O Reward: {o_reward}', True, TEXT_COLOR, BACKGROUND_COLOR)
    text_rect = text.get_rect()
    text_rect.bottomleft = (10, HEIGHT - 40)
    screen.blit(text, text_rect)


def draw_message(msg):
    text = font.render(msg, True, TEXT_COLOR, BACKGROUND_COLOR)
    text_rect = text.get_rect()
    text_rect.center = (WIDTH // 2, HEIGHT // 2)
    screen.blit(text, text_rect)


def game_over(winner):
    if winner == 'X':
        draw_message('X wins')

    elif winner == 'O':
        draw_message('O wins')


# Check X condition first so if X wins on the 50 turn, there is not a tie process

def check_win_conditions_x():
    global x_x_pos, x_y_pos, o_x_pos, o_y_pos, match_over
    if x_y_pos == o_y_pos and x_x_pos == o_x_pos:
        match_over = True
        return True
    return False


# Check to make sure O didn't enter the X cell and suicide, then check if it has out-lasted X
def check_win_conditions_o():
    global turn_counter, match_over, x_x_pos, x_y_pos, o_x_pos, o_y_pos
    if x_y_pos == o_y_pos and x_x_pos == o_x_pos:
        match_over = True
        return False
    if turn_counter == 50:
        match_over = True
        return True
    return False


def calculate_reward_x(state, state_next, action_blocked, winner):
    global GRID_SIZE

    x_raw = np.where(state == 1)[0]
    o_raw = np.where(state == 2)[0]

    x_raw_next = np.where(state_next == 1)[0]
    o_raw_next = np.where(state_next == 2)[0]

    x_pos = (x_raw % GRID_SIZE, x_raw // GRID_SIZE)
    o_pos = (o_raw % GRID_SIZE, o_raw // GRID_SIZE)

    x_pos_next = (x_raw_next % GRID_SIZE, x_raw_next // GRID_SIZE)
    o_pos_next = (o_raw_next % GRID_SIZE, o_raw_next // GRID_SIZE)

    if winner:
        return 100
    else:
        reward = 0
        if action_blocked:
            reward -= 1
        dist_prev = abs(x_pos[0] - o_pos[0]) + abs(x_pos[1] - o_pos[1])
        dist_curr = abs(x_pos_next[0] - o_pos_next[0]) + abs(x_pos_next[1] - o_pos_next[1])
        return reward + 10 if dist_curr < dist_prev else reward - 1


def calculate_reward_o(state, state_next, action_blocked, winner):
    global GRID_SIZE

    x_raw = np.where(state == 1)[0]
    o_raw = np.where(state == 2)[0]

    x_raw_next = np.where(state_next == 1)[0]
    o_raw_next = np.where(state_next == 2)[0]

    x_pos = (x_raw % GRID_SIZE, x_raw // GRID_SIZE)
    o_pos = (o_raw % GRID_SIZE, o_raw // GRID_SIZE)

    x_pos_next = (x_raw_next % GRID_SIZE, x_raw_next // GRID_SIZE)
    o_pos_next = (o_raw_next % GRID_SIZE, o_raw_next // GRID_SIZE)

    if winner:
        return 100
    else:
        global turn_counter
        reward = turn_counter * 0.1
        if action_blocked:
            reward -= 1
        dist_prev = abs(x_pos[0] - o_pos[0]) + abs(x_pos[1] - o_pos[1])
        dist_curr = abs(x_pos_next[0] - o_pos_next[0]) + abs(x_pos_next[1] - o_pos_next[1])
        return reward + 1 if dist_curr > dist_prev else reward


# returns true if action was blocked, else returns false
def do_action_x(action):
    global x_x_pos, x_y_pos, GRID_SIZE
    if action == 0 and x_y_pos != 0:
        x_y_pos = x_y_pos - 1
        return False
    elif action == 1 and x_y_pos != GRID_SIZE - 1:
        x_y_pos = x_y_pos + 1
        return False
    elif action == 2 and x_x_pos != 0:
        x_x_pos = x_x_pos - 1
        return False
    elif action == 3 and x_x_pos != GRID_SIZE - 1:
        x_x_pos = x_x_pos + 1
        return False
    elif action == 4:
        return False
    return True


def do_action_o(action):
    global o_x_pos, o_y_pos, GRID_SIZE
    if action == 0 and o_y_pos != 0:
        o_y_pos = o_y_pos - 1
        return False
    elif action == 1 and o_y_pos != GRID_SIZE - 1:
        o_y_pos = o_y_pos + 1
        return False
    elif action == 2 and o_x_pos != 0:
        o_x_pos = o_x_pos - 1
        return False
    elif action == 3 and o_x_pos != GRID_SIZE - 1:
        o_x_pos = o_x_pos + 1
        return False
    elif action == 4:
        return False
    return True


def match_reset():
    global turn_counter, match_over, x_x_pos, x_y_pos, o_x_pos, o_y_pos, x_reward, o_reward
    x_x_pos = random.randint(0, GRID_SIZE - 1)
    x_y_pos = random.randint(0, GRID_SIZE - 1)
    o_x_pos = random.randint(0, GRID_SIZE - 1)
    o_y_pos = random.randint(0, GRID_SIZE - 1)
    turn_counter = 0
    match_over = False
    x_reward = 0
    o_reward = 0


def select_action(state, policy_net, epsilon, n_actions):
    # If a randomly chosen number is less than epsilon, select a random action
    sample = random.random()
    if sample < epsilon:
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0)
            q_values = policy_net(state)
            print("choosing polocy action")
            return int(torch.argmax(q_values).item())  # Convert to Python int
    else:
        # print("random")
        return random.choice(range(n_actions))  # Return a Python int


def get_epsilon(episode):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * episode / EPSILON_DECAY)


# memory frames

memory_x = ReplayMemory(10000)
memory_o = ReplayMemory(10000)


# Initial Loop, without networks

def initial_loop():
    global turn_counter, x_reward, o_reward

    a_x = random.choice(range(len(actions)))
    a_o = random.choice(range(len(actions)))

    state_x = calculate_state()
    state_o = calculate_state()

    x_blocked = do_action_x(a_x)
    x_win = check_win_conditions_x()
    if x_win:
        state_x_next = None
    else:
        state_x_next = calculate_state()
    x_curr_rew = calculate_reward_x(state_x, state_x_next, x_blocked, x_win)
    x_reward += x_curr_rew

    o_blocked = do_action_o(a_o)
    o_win = check_win_conditions_o()
    if o_win:
        state_o_next = None
    else:
        state_o_next = calculate_state()
    o_curr_rew = calculate_reward_o(state_o, state_o_next, o_blocked, o_win)
    o_reward += o_curr_rew

    memory_x.push(state_x, a_x, state_x_next, x_curr_rew)
    memory_o.push(state_o, a_o, state_o_next, o_curr_rew)

    if match_over:
        match_reset()

    turn_counter += 1


# instantiate the neural networks

policy_net_x = DQN(400, len(actions))
target_net_x = DQN(400, len(actions))
policy_net_o = DQN(400, len(actions))
target_net_o = DQN(400, len(actions))

target_net_x.load_state_dict(policy_net_x.state_dict())
target_net_o.load_state_dict(policy_net_o.state_dict())

target_net_x.eval()
target_net_o.eval()

optimizer_x = optim.Adam(policy_net_x.parameters(), lr=LR)
optimizer_o = optim.Adam(policy_net_o.parameters(), lr=LR)


# loop with networks

def network_loop(episodes):
    global turn_counter, x_reward, o_reward

    state_x = calculate_state()
    state_o = calculate_state()

    eps = get_epsilon(episodes)
    a_x = select_action(state_x, policy_net_x, eps, len(actions))
    a_o = select_action(state_o, policy_net_o, eps, len(actions))

    x_blocked = do_action_x(a_x)
    x_win = check_win_conditions_x()
    if x_win:
        state_x_next = None
    else:
        state_x_next = calculate_state()
    x_curr_rew = calculate_reward_x(state_x, state_x_next, x_blocked, x_win)
    x_reward += x_curr_rew

    o_blocked = do_action_o(a_o)
    o_win = check_win_conditions_o()
    if o_win:
        state_o_next = None
    else:
        state_o_next = calculate_state()
    o_curr_rew = calculate_reward_o(state_o, state_o_next, o_blocked, o_win)
    o_reward += o_curr_rew

    memory_x.push(state_x, a_x, state_x_next, x_curr_rew)
    memory_o.push(state_o, a_o, state_o_next, o_curr_rew)

    if match_over:
        match_reset()

    turn_counter += 1

    print("optimizing")
    optimize_model(memory_x, policy_net_x, target_net_x, optimizer_x)
    optimize_model(memory_o, policy_net_o, target_net_o, optimizer_o)


# the main game loop

episode = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w and x_y_pos != 0:
                x_y_pos = x_y_pos - 1
            elif event.key == pygame.K_s and x_y_pos != GRID_SIZE - 1:
                x_y_pos = x_y_pos + 1
            elif event.key == pygame.K_a and x_x_pos != 0:
                x_x_pos = x_x_pos - 1
            elif event.key == pygame.K_d and x_x_pos != GRID_SIZE - 1:
                x_x_pos = x_x_pos + 1
            elif event.key == pygame.K_UP and o_y_pos != 0:
                o_y_pos = o_y_pos - 1
            elif event.key == pygame.K_DOWN and o_y_pos != GRID_SIZE - 1:
                o_y_pos = o_y_pos + 1
            elif event.key == pygame.K_LEFT and o_x_pos != 0:
                o_x_pos = o_x_pos - 1
            elif event.key == pygame.K_RIGHT and o_x_pos != GRID_SIZE - 1:
                o_x_pos = o_x_pos + 1
            elif event.key == pygame.K_SPACE:
                match_over = True

    while episode < 1000:
        initial_loop()
        episode += 1


    network_loop(episode)

    screen.fill((0, 0, 0))

    # Update and draw game state here
    draw_grid()
    draw_o(o_x_pos, o_y_pos)
    draw_x(x_x_pos, x_y_pos)
    draw_turn_counter()
    draw_x_reward()
    draw_o_reward()

    pygame.display.flip()
    clock.tick(60)
    episode += 1

    if episode % 100 == 0:
        target_net_x.load_state_dict(policy_net_x.state_dict())
        target_net_o.load_state_dict(policy_net_o.state_dict())

pygame.quit()
sys.exit()
