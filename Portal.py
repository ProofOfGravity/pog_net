import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np
from DQG import QTrainer, Linear_QNet
from DDQN import DuelingDQN

pygame.init()

## working

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

GRID_SIZE = 20
CELL_SIZE = 30
WIDTH = HEIGHT = CELL_SIZE * GRID_SIZE
BACKGROUND_COLOR = (0, 0, 0)
GRID_COLOR = (128, 128, 128)
X_COLOR = (255, 0, 0)
O_COLOR = (0, 255, 0)
PORTAL_COLOR = (0, 0, 255)  # New color for portal
TEXT_COLOR = (255, 255, 255)
FONT_SIZE = 30
FPS = 60
NUM_BLOCKS = GRID_SIZE  # Number of blocks equals to GRID_SIZE for the middle barrier

BUFFER_SIZE = 50000
REPLAY_INITIAL_SIZE = 20000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 1000

GAMMA = 0.99

font = pygame.font.Font(None, FONT_SIZE)

class XOGame:

    def __init__(self):

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('X and O Game')
        self.clock = pygame.time.Clock()

        self.frame_iteration = 0

        self.x_pos = Point(0, 0)
        self.x_pos_previous = self.x_pos

        self.reward_buffer_x = deque(maxlen=100)

        self.reward_x_total = 0
        self.reward_o_total = 0

        self.replay_buffer_x = deque(maxlen=BUFFER_SIZE)
        self.replay_buffer_o = deque(maxlen=BUFFER_SIZE)

        self.blocks = []
        self.portal = []  # New list for portal points

        self.reset()

    def reset(self):

        # X and O spawn at random points on opposite sides of the middle dividing line
        self.x_pos = Point(random.randint(0, GRID_SIZE // 2 - 1), random.randint(0, GRID_SIZE - 1))
        self.o_pos = Point(random.randint(GRID_SIZE // 2 + 1, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

        self.frame_iteration = 0

        self.reward_x_total = 0
        self.reward_o_total = 0

        # Create a barrier directly down the middle of the grid
        self.blocks = [Point(GRID_SIZE // 2, i) for i in range(GRID_SIZE)]

        # Create portal points
        self.portal = []
        while len(self.portal) < 1:
            portal_pos = Point(random.randint(0, GRID_SIZE // 2 - 1), random.randint(0, GRID_SIZE - 1))
            if portal_pos != self.x_pos and portal_pos != self.o_pos and portal_pos not in self.blocks and portal_pos not in self.portal:
                self.portal.append(portal_pos)
        while len(self.portal) < 2:
            portal_pos = Point(random.randint(GRID_SIZE // 2 + 1, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if portal_pos != self.x_pos and portal_pos != self.o_pos and portal_pos not in self.blocks and portal_pos not in self.portal:
                self.portal.append(portal_pos)

    def update_ui(self):

        # clear screen
        self.screen.fill((0, 0, 0))

        # draw grid
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WIDTH, y))

        # draw X
        text = font.render('X', True, X_COLOR, BACKGROUND_COLOR)
        text_rect = text.get_rect()
        text_rect.center = (self.x_pos.x * CELL_SIZE + CELL_SIZE // 2, self.x_pos.y * CELL_SIZE + CELL_SIZE // 2)
        self.screen.blit(text, text_rect)

        # draw O
        text = font.render('O', True, O_COLOR, BACKGROUND_COLOR)
        text_rect = text.get_rect()
        text_rect.center = (self.o_pos.x * CELL_SIZE + CELL_SIZE // 2, self.o_pos.y * CELL_SIZE + CELL_SIZE // 2)
        self.screen.blit(text, text_rect)

        # draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, (128, 128, 128),
                             pygame.Rect(block.x * CELL_SIZE, block.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # draw portals
        for portal in self.portal:
            pygame.draw.rect(self.screen, PORTAL_COLOR,
                             pygame.Rect(portal.x * CELL_SIZE, portal.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # draw X reward
        text = font.render(f'X Reward: {self.reward_x_total}', True, TEXT_COLOR, BACKGROUND_COLOR)
        text_rect = text.get_rect()
        text_rect.bottomleft = (10, HEIGHT - 10)
        self.screen.blit(text, text_rect)

        # draw O reward
        text = font.render(f'O Reward: {self.reward_o_total}', True, TEXT_COLOR, BACKGROUND_COLOR)
        text_rect = text.get_rect()
        text_rect.bottomleft = (10, HEIGHT - 40)
        self.screen.blit(text, text_rect)

        pygame.display.flip()

    def play_step(self, action_x, action_o):

        # 0 Up, 1 Down, 2 Left, 3 Right, 4 No-Op

        # First get the initial state
        state_x_initial = self.calculate_state_x()

        # 1. Check if game is over, if so, report Winner/Loser and reset
        game_over, winner, loser, x_reward, o_reward = self.check_move(action_x, action_o)
        self.reward_x_total += x_reward

        # 2. Move actors
        self.move(action_x, action_o)

        # 3. Collect new state (does not matter if game over, this state will be thrown away if that is the case)
        state_x_next = self.calculate_state_x()

        # 4. Add transition to replay buffer
        transition = (state_x_initial, action_x, x_reward, game_over, state_x_next)
        self.replay_buffer_x.append(transition)

        # 5. If game over, then print winner and reset environment
        if game_over:
            print(f'Game Over! {winner} wins, {loser} loses')
            self.reward_buffer_x.append(self.reward_x_total)
            self.reset()
            return

        # 6. Update step count
        self.frame_iteration += 1

        # 7. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 8. Update UI
        self.update_ui()
        self.clock.tick(FPS)

    def initialize_replay_memory(self):
        for _ in range(REPLAY_INITIAL_SIZE):
            # First get the initial state
            state_x_initial = self.calculate_state_x()

            # 0 Up, 1 Down, 2 Left, 3 Right
            x_move = random.randint(0, 4)
            o_move = 4

            # 1. Check if game is over, if so, report Winner/Loser and reset
            game_over, winner, loser, x_reward, o_reward = self.check_move(x_move, o_move)

            # 2. Move actors
            self.move(x_move, o_move)

            # 3. Collect new state (does not matter if game over, this state will be thrown away if that is the case)
            state_x_next = self.calculate_state_x()

            # 4. Add transition to replay buffer
            transition = (state_x_initial, x_move, x_reward, game_over, state_x_next)
            self.replay_buffer_x.append(transition)

            # 5. If game over, then print winner and reset environment
            if game_over:
                print(f'Game Over! {winner} wins, {loser} loses')
                self.reset()

    def move(self, action_x, action_o):

        # 0 Up, 1 Down, 2 Left, 3 Right, 4 No-Op
        # Move X actor
        if action_x == 0:
            self.x_pos.y -= 1
        elif action_x == 1:
            self.x_pos.y += 1
        elif action_x == 2:
            self.x_pos.x -= 1
        elif action_x == 3:
            self.x_pos.x += 1

        # Move O actor
        if action_o == 0:
            self.o_pos.y -= 1
        elif action_o == 1:
            self.o_pos.y += 1
        elif action_o == 2:
            self.o_pos.x -= 1
        elif action_o == 3:
            self.o_pos.x += 1

        # Check if X is on a portal, if so, transport to the other side
        if self.x_pos == self.portal[0]:
            self.x_pos = Point(self.portal[1].x, self.portal[1].y)
        elif self.x_pos == self.portal[1]:
            self.x_pos = Point(self.portal[0].x, self.portal[0].y)

    def check_move(self, action_x, action_o):
        game_over = False
        winner = None
        loser = None
        reward_x = 0
        reward_o = 0

        # 0 Up, 1 Down, 2 Left, 3 Right, 4 No-Op
        # Calculate the new positions
        if action_x == 0:
            new_x_pos = Point(self.x_pos.x, self.x_pos.y - 1)
        elif action_x == 1:
            new_x_pos = Point(self.x_pos.x, self.x_pos.y + 1)
        elif action_x == 2:
            new_x_pos = Point(self.x_pos.x - 1, self.x_pos.y)
        elif action_x == 3:
            new_x_pos = Point(self.x_pos.x + 1, self.x_pos.y)
        elif action_x == 4:
            new_x_pos = Point(self.x_pos.x, self.x_pos.y)

        if action_o == 0:
            new_o_pos = Point(self.o_pos.x, self.o_pos.y - 1)
        elif action_o == 1:
            new_o_pos = Point(self.o_pos.x, self.o_pos.y + 1)
        elif action_o == 2:
            new_o_pos = Point(self.o_pos.x - 1, self.o_pos.y)
        elif action_o == 3:
            new_o_pos = Point(self.o_pos.x + 1, self.o_pos.y)
        elif action_o == 4:
            new_o_pos = Point(self.o_pos.x, self.o_pos.y)

        # Check if X is going off the grid
        if not (0 <= new_x_pos.x < GRID_SIZE and 0 <= new_x_pos.y < GRID_SIZE):
            game_over = True
            winner = 'O'
            loser = 'X'
        # Check if X is moving into O's space
        elif new_x_pos.x == self.o_pos.x and new_x_pos.y == self.o_pos.y:
            game_over = True
            winner = 'X'
            loser = 'O'
        # Check if O is going off the grid
        elif not (0 <= new_o_pos.x < GRID_SIZE and 0 <= new_o_pos.y < GRID_SIZE):
            game_over = True
            winner = 'X'
            loser = 'O'
        # Check if its turn 50, if so O wins
        elif self.frame_iteration == 50:
            game_over = True
            winner = 'O'
            loser = 'X'

        # Check if X hits a block
        if new_x_pos in self.blocks:
            game_over = True
            winner = 'O'
            loser = 'X'

        if game_over and winner == 'X':
            reward_x += 100
        elif game_over and winner == 'O':
            reward_x -= 100
        else:

            if new_x_pos == self.x_pos_previous:
                reward_x -= 1.5

            # X is on the side it spawned on
            if self.x_pos.x < GRID_SIZE // 2:
                dist_prev_portal = abs(self.x_pos.x - self.portal[0].x) + abs(self.x_pos.y - self.portal[0].y)
                dist_curr_portal = abs(new_x_pos.x - self.portal[0].x) + abs(new_x_pos.y - self.portal[0].y)
                if dist_curr_portal < dist_prev_portal:
                    reward_x = reward_x + 1 if dist_curr_portal < dist_prev_portal else reward_x - 1
            # X has teleported to the other side
            else:
                dist_prev = abs(self.x_pos.x - self.o_pos.x) + abs(self.x_pos.y - self.o_pos.y)
                dist_curr = abs(new_x_pos.x - self.o_pos.x) + abs(new_x_pos.y - self.o_pos.y)
                reward_x = reward_x + 1 if dist_curr < dist_prev else reward_x - 1

        reward_x -= 0.1

        self.x_pos_previous = self.x_pos

        return game_over, winner, loser, reward_x, reward_o

    def calculate_state_x(self):

        # state will consist of 8 values, that will essentially be booleans
        # The first 4 describe if there is a danger in a direction, such as a wall or the dividing line
        # The 2nd group of 4 describe the direction of the objective (For X, the direction to the portal or O)

        state_x = np.zeros(8, dtype=int)

        # Check for danger in all four directions, meaning going off grid or hitting block or the dividing line
        if self.x_pos.y == 0 or Point(self.x_pos.x, self.x_pos.y - 1) in self.blocks or self.x_pos.x == GRID_SIZE // 2:
            state_x[0] = 1
        if self.x_pos.y == GRID_SIZE - 1 or Point(self.x_pos.x,
                                                  self.x_pos.y + 1) in self.blocks or self.x_pos.x == GRID_SIZE // 2:
            state_x[1] = 1
        if self.x_pos.x == 0 or Point(self.x_pos.x - 1, self.x_pos.y) in self.blocks or self.x_pos.x == GRID_SIZE // 2:
            state_x[2] = 1
        if self.x_pos.x == GRID_SIZE - 1 or Point(self.x_pos.x + 1,
                                                  self.x_pos.y) in self.blocks or self.x_pos.x == GRID_SIZE // 2:
            state_x[3] = 1

        # These describe the general direction to the portal or O
        if self.x_pos.x < GRID_SIZE // 2:
            if self.x_pos.y > self.portal[0].y:
                state_x[4] = 1
            if self.x_pos.y < self.portal[0].y:
                state_x[5] = 1
            if self.x_pos.x > self.portal[0].x:
                state_x[6] = 1
            if self.x_pos.x < self.portal[0].x:
                state_x[7] = 1
        else:
            if self.x_pos.y > self.o_pos.y:
                state_x[4] = 1
            if self.x_pos.y < self.o_pos.y:
                state_x[5] = 1
            if self.x_pos.x > self.o_pos.x:
                state_x[6] = 1
            if self.x_pos.x < self.o_pos.x:
                state_x[7] = 1

        return state_x

    def learn_without_display(self, iterations):
        for _ in range(iterations):
            epsilon = np.interp(x_trainer.step_count, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

            rnd_sample = random.random()

            if rnd_sample <= epsilon:
                a_x = random.randint(0, 4)
            else:
                a_x = x_net.act(game.calculate_state_x())

            # First get the initial state
            state_x_initial = self.calculate_state_x()

            # 1. Check if game is over, if so, report Winner/Loser and reset
            game_over, winner, loser, x_reward, o_reward = self.check_move(a_x, 4)
            self.reward_x_total += x_reward

            # 2. Move actors
            self.move(a_x, 4)

            # 3. Collect new state (does not matter if game over, this state will be thrown away if that is the case)
            state_x_next = self.calculate_state_x()

            # 4. Add transition to replay buffer
            transition = (state_x_initial, a_x, x_reward, game_over, state_x_next)
            self.replay_buffer_x.append(transition)

            # 5. If game over, then print winner and reset environment
            if game_over:
                print(f'Game Over! {winner} wins, {loser} loses')
                self.reward_buffer_x.append(self.reward_x_total)
                self.reset()

            # 6. Update step count
            self.frame_iteration += 1

            x_trainer.train_step(game.replay_buffer_x, 100, GAMMA)


game = XOGame()

# x_net = Linear_QNet(8, 64, 5)  # Update the input size to 8
x_net = DuelingDQN(8, 5)

x_trainer = QTrainer(x_net, 0.0005, GAMMA)

game.initialize_replay_memory()

#game.learn_without_display(10000)

while True:

    epsilon = np.interp(x_trainer.step_count, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        a_x = random.randint(0, 4)
    else:
        a_x = x_net.act(game.calculate_state_x())

    game.play_step(a_x, 4)

    x_trainer.train_step(game.replay_buffer_x, 500, GAMMA)

    # if np.mean(game.reward_x_total) >= 60:
    #     FPS = 10
