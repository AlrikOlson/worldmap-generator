# src/core/game.py
import pygame
import threading
import numpy as np
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
        self.render_surface = self.world_surface.copy()  # Separate surface for rendering
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
            # Copy world_surface to render_surface to avoid locking issues during blit
            self.render_surface = self.world_surface.copy()
            self.screen.blit(self.render_surface, (0, 0))  # Blit the world surface to the screen
        self.render_progress()
        pygame.display.flip()

    def render_progress(self):
        if self.progress < 100:
            font = pygame.font.Font(None, 36)
            text = font.render(f'Generating World: {self.progress:.1f}%', True, (255, 255, 255))
            self.screen.blit(text, (10, 10))

    def update_world_surface(self):
        try:
            # Ensure the `self.world` tensor matches the expected shape
            if self.world.shape != (1280, 720):
                raise ValueError("Generated world shape does not match surface dimensions")

            # Convert the `self.world` tensor to a numpy array and ensure it's properly shaped
            world_np = self.world.cpu().numpy()
            
            # Initialize color array with the shape (width, height, 3)
            color_array = np.zeros((1280, 720, 3), dtype=np.uint8)  # Note: width, height, 3
            
            for x in range(1280):
                for y in range(720):
                    value = world_np[x, y]
                    color = self.get_color(value)
                    if not all(0 <= c <= 255 for c in color):  # Ensure color values are within valid range
                        raise ValueError(f"Invalid color value {color} at ({x}, {y})")
                    color_array[x, y] = color  # Note: access pattern corrected for debugging

            print(f"world_surface size: {self.world_surface.get_size()}")
            print(f"color_array.shape: {color_array.shape}")

            # Efficiently blit the color array to the world surface using PixelArray
            pixel_array = pygame.PixelArray(self.world_surface)
            for x in range(1280):
                for y in range(720):
                    pixel_array[x, y] = tuple(color_array[x, y])
            del pixel_array  # Unlock the surface by deleting the PixelArray reference
        except Exception as e:
            # Output detailed information about the exception and the relevant data
            print(f"Exception occurred: {e}")
            print(f"self.world.shape: {self.world.shape}")
            print(f"world_np.shape: {world_np.shape} -- dtype: {world_np.dtype}")
            print(f"color_array.shape: {color_array.shape} -- dtype: {color_array.dtype}")
            if 'color' in locals():
                print(f"Invalid color value encountered: {color}")
            raise  # Re-raise the exception after printing the details

    def get_color(self, value):
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
