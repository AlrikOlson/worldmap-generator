# src/core/game.py
import pygame
from procedural.map_generator import MapGenerator  # Relative import

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True
        self.map_gen = MapGenerator(1280, 720)  # Example dimensions

    def run(self):
        self.world = self.map_gen.generate()
        
        while self.running:
            self.handle_events()
            self.update()
            self.render()

            self.clock.tick(60)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        # Update game state
        pass

    def render(self):
        self.screen.fill((0, 0, 0))

        for x in range(self.world.shape[0]):
            for y in range(self.world.shape[1]):
                value = self.world[x][y]
                color = self.get_color(value)
                self.screen.set_at((x, y), color)

        pygame.display.flip()

    def get_color(self, value):
        if value < 0.3:
            return (0, 0, 128) # Deep water
        elif 0.3 <= value < 0.4:
            return (0, 128, 255) # Shallow water
        elif 0.4 <= value < 0.5:
            return (240, 230, 140) # Beach
        elif 0.5 <= value < 0.7:
            return (34, 139, 34) # Grass
        elif 0.7 <= value < 0.85:
            return (160, 82, 45) # Mountain
        else:
            return (255, 250, 250) # Snow

# Usage example
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    game = Game(screen)
    game.run()
    pygame.quit()
