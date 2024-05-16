# src/core/game.py
import pygame
import threading
from procedural.map_generator import MapGenerator  # Relative import

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True
        self.map_gen = MapGenerator(1280, 720, scale=200.0, octaves=6)  # Example dimensions and parameters
        self.world = None  # Initial map is None until generated
        self.progress = 0  # Progress indicator
        self.world_surface = pygame.Surface((1280, 720))  # Surface for the map
        self.generate_world()  # Start generating the world

    def generate_world(self):
        self.progress = 0
        threading.Thread(target=self.run_map_generation).start()  # Run map generation in a separate thread

    def run_map_generation(self):
        self.world = self.map_gen.generate(self.update_progress)
        self.update_world_surface()

    def update_progress(self, progress):
        self.progress = progress

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Check if the spacebar is pressed
                    self.generate_world()  # Regenerate the map with a new seed
    
    def update(self):
        # Update game state
        pass
    
    def render(self):
        self.screen.fill((0, 0, 0))
        if self.world is not None:
            self.screen.blit(self.world_surface, (0, 0))  # Blit the world surface to the screen
        self.render_progress()
        pygame.display.flip()

    def render_progress(self):
        if self.progress < 100:
            font = pygame.font.Font(None, 36)
            text = font.render(f'Generating World: {self.progress:.1f}%', True, (255, 255, 255))
            self.screen.blit(text, (10, 10))

    def update_world_surface(self):
        array = pygame.surfarray.pixels3d(self.world_surface)  # Get the pixel array of the surface
        for x in range(self.world.shape[0]):
            for y in range(self.world.shape[1]):
                value = self.world[x][y]
                color = self.get_color(value, x, y)
                array[x][y] = color  # Set the pixel color

    def get_color(self, value, x, y):
        # Adjusted biomes based on elevation
        if value < 0.15:
            return (0, 0, 128)  # Deep water
        elif 0.15 <= value < 0.25:
            return (0, 128, 255)  # Shallow water
        elif 0.25 <= value < 0.3:
            return (240, 230, 140)  # Beach
        elif 0.3 <= value < 0.55:
            return (34, 139, 34)  # Grass
        elif 0.55 <= value < 0.75:
            return (60, 179, 113)  # Forest
        elif 0.75 <= value < 0.85:
            return (160, 82, 45)  # Mountain
        else:
            return (255, 250, 250)  # Snow

# Usage example
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    game = Game(screen)
    game.run()
    pygame.quit()
