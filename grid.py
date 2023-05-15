import pygame
import math
import pickle


class Grid:
    def __init__(self, num_cells_x, num_cells_y, rect: pygame.rect):
        self.window = pygame.display.get_surface()
        self.surface = pygame.Surface((rect.width, rect.height))
        self.window_size = (rect.width, rect.height)
        self.num_cells_x = num_cells_x
        self.num_cells_y = num_cells_y
        self.rect_dim_x = self.window_size[0] / num_cells_x
        self.rect_dim_y = self.window_size[1] / num_cells_y

        self.rect = rect

        self.rect_list = []
        self.is_active_list = []

        self.is_editable = True

        for y in range(self.num_cells_y):
            for x in range(self.num_cells_x):
                self.rect_list.append(
                    pygame.Rect(x * self.rect_dim_x, y * self.rect_dim_y, self.rect_dim_x, self.rect_dim_y))
                self.is_active_list.append(True)

    def draw_rect_grid(self):
        self.surface.fill((0, 0, 0))
        for i in range(len(self.rect_list)):
            if self.is_active_list[i]:
                pygame.draw.rect(self.surface, (200, 180, 23), self.rect_list[i], 2)
            else:
                pygame.draw.rect(self.surface, (200, 180, 23), self.rect_list[i], 10)
        self.window.blit(self.surface, self.rect)

    def convert_screen_to_grid(self, x, y):
        x = math.floor((x - self.rect.left) / self.rect_dim_x)
        y = math.floor((y - self.rect.top) / self.rect_dim_y)
        return x, y

    def turn_cell_off(self, x, y):
        self.is_active_list[(y * self.num_cells_x) + x] = False

    def turn_cell_on(self, x, y):
        self.is_active_list[(y * self.num_cells_x) + x] = True

    def update_grid(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.is_editable:
            if self.rect.collidepoint(event.pos[0], event.pos[1]):
                x, y = self.convert_screen_to_grid(event.pos[0], event.pos[1])

                if self.is_active_list[(y * self.num_cells_x) + x]:
                    self.turn_cell_off(x, y)
                    print("turning cell off")
                else:
                    self.turn_cell_on(x, y)
                    print("turning cell on")

    def save_grid(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.is_active_list, f)

    def open_grid(self, file_name):
        with open(file_name, 'rb') as f:
            self.is_active_list = pickle.load(f)

    def turn_on_edits(self):
        self.is_editable = True