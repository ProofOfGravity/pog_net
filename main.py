import pygame
import grid



class App:
    def __init__(self, app_name, width, height, flags=0):
        pygame.display.set_caption(app_name)
        self.surface = pygame.display.set_mode((width, height), flags)
        self.screen_rect = self.surface.get_rect()
        self.clock = pygame.time.Clock()
        self.is_running = False
        self.delta = 0
        self.fps = 60


        rect = pygame.Rect(0, 0, (self.screen_rect.width*0.75), self.screen_rect.height)
        self.grid = grid.Grid(10, 10, rect)



    def run(self):


        self.is_running = True
        while self.is_running:

            self.surface.fill((0, 0, 0))

            event_list = pygame.event.get()
            for event in event_list:
                if event.type == pygame.QUIT:
                    self.is_running = False
                self.grid.update_grid(event)

            self.grid.draw_rect_grid()


            pygame.display.flip()
            self.delta = self.clock.tick(self.fps)


if __name__ == "__main__":
    pygame.init()
    app = App("POG Test", 800, 600)
    app.run()
    pygame.quit()
