# src/core/game.py
import pygame
import threading
import numpy as np
import torch
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
            print("Starting update_world_surface")
            print(f"self.world dtype: {self.world.dtype}")
            print(f"self.world shape: {self.world.shape}")
            if self.world.shape != (1280, 720):
                raise ValueError("Generated world shape does not match surface dimensions")

            # Convert the `self.world` tensor to a numpy array and ensure it's properly shaped
            world_np = self.world.cpu().numpy()
            print(f"world_np shape: {world_np.shape}, dtype: {world_np.dtype}")
            
            color_array = self.generate_color_array(world_np)
            print(f"color_array shape: {color_array.shape}, dtype: {color_array.dtype}")

            # Efficiently blit the color array to the world surface
            pygame.surfarray.blit_array(self.world_surface, color_array)
        except Exception as e:
            # Output detailed information about the exception and the relevant data
            print(f"Exception occurred: {e}")
            if hasattr(self, 'world'):
                print(f"self.world.shape: {self.world.shape}")
            print(f"world_np.shape: {world_np.shape} -- dtype: {world_np.dtype}")
            if 'color_array' in locals():
                print(f"color_array.shape: {color_array.shape} -- dtype: {color_array.dtype}")
            raise  # Re-raise the exception after printing the details

    def generate_color_array(self, world_np):
        print("Starting generate_color_array")
        
        # Convert the numpy array to a torch tensor and move it to the GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_tensor = torch.from_numpy(world_np).to(device)

        # Define thresholds and colors, adjusted for smoother transitions
        thresholds = torch.tensor([0.2, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85], device=device)
        colors = torch.tensor([
            [0, 0, 128],     # Deep water
            [0, 128, 255],   # Shallow water
            [240, 230, 140], # Beach
            [34, 139, 34],   # Grass
            [60, 179, 113],  # Forest
            [160, 82, 45],   # Mountain
            [255, 250, 250], # Snow
            [128, 128, 128], # Rocky shoreline (for GPU-only implementation, could be managed here or later)
        ], dtype=torch.float32, device=device)
        
        # Initialize a tensor for the color array on the GPU
        color_array = torch.zeros((1280, 720, 3), dtype=torch.uint8, device=device)

        # Function to blend colors
        def blend_colors(color1, color2, factor):
            return (color1 * (1 - factor) + color2 * factor).to(dtype=torch.uint8)

        # Function to smooth elevation
        def smooth_elevation(elevations, kernel_size=3):
            padding = kernel_size // 2
            padded_elevations = torch.nn.functional.pad(elevations.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='replicate').squeeze()
            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) / (kernel_size * kernel_size)
            smoothed = torch.nn.functional.conv2d(padded_elevations.unsqueeze(0).unsqueeze(0), kernel).squeeze()
            return smoothed

        smoothed_tensor = smooth_elevation(world_tensor, 3)

        for i in range(len(thresholds) - 1):
            lower_threshold = thresholds[i]
            upper_threshold = thresholds[i + 1]
            mask = (smoothed_tensor >= lower_threshold) & (smoothed_tensor < upper_threshold)
            blend_factor = (smoothed_tensor[mask] - lower_threshold) / (upper_threshold - lower_threshold)
            color1 = colors[i]
            color2 = colors[i+1]
            blended_colors = blend_colors(color1, color2, blend_factor.unsqueeze(-1))
            color_array[mask] = blended_colors

        # Handle last threshold separately (snow color)
        mask = smoothed_tensor >= thresholds[-1]
        color_array[mask] = colors[-1].to(dtype=torch.uint8)

        # Move the color array back to the CPU
        color_array = color_array.cpu().numpy()

        print(f"Final color_array shape: {color_array.shape}, dtype: {color_array.dtype}")

        return color_array

# Usage example
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    game = Game(screen)
    game.run()
    pygame.quit()
