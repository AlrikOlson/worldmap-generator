import pygame
import threading
import numpy as np
import torch
from procedural.map_generator import MapGenerator
from ui.button import Button
import random
from scipy.ndimage import measurements

class Game:
    def __init__(self, screen, map_width, map_height):
        self.screen = screen
        self.map_width = map_width
        self.map_height = map_height
        self.clock = pygame.time.Clock()
        self.running = True
        self.map_gen = MapGenerator(map_width, map_height, scale=400.0, octaves=48)
        self.world = None
        self.progress = 0
        self.world_surface = pygame.Surface((map_width, map_height))
        self.render_surface = self.world_surface.copy()
        self.continent_names = []
        self.labels = None
        self.generate_world()

        button_width, button_height = 100, 50
        self.quit_button = Button(
            rect=(map_width - button_width - 10, map_height - button_height - 10, button_width, button_height),
            color=(150, 0, 0),
            hover_color=(200, 0, 0),
            text='Quit',
            font=pygame.font.Font(None, 36),
            text_color=(255, 255, 255),
            callback=self.quit_game
        )

    def generate_world(self):
        self.progress = 0
        self.continent_names = []
        threading.Thread(target=self.run_map_generation).start()

    def run_map_generation(self):
        self.world = self.map_gen.generate(self.update_progress)
        self.label_continents()
        self.update_world_surface()

    def update_progress(self, progress):
        self.progress = progress

    def label_continents(self):
        world_np = self.world.cpu().numpy()
        self.labels, num_labels = self.label_connected_regions(world_np)

        potential_names = [
            "Atlantis", "Elysium", "Arcadia", "Erewhon", "Shangri-La",
            "Avalon", "El Dorado", "Valhalla", "Utopia", "Narnia"
        ]
        random.shuffle(potential_names)

        self.continent_names = []
        for label in range(1, num_labels + 1):
            area = np.sum(self.labels == label)
            if area > 10000:  # Only consider regions larger than a certain threshold
                # Make sure the region isn't primarily water
                region_elevations = world_np[self.labels == label]
                if np.mean(region_elevations) > 0.011:  # Exclude regions with low avg elevation (water)
                    if len(self.continent_names) < len(potential_names):
                        name = potential_names[len(self.continent_names)]
                    else:
                        name = f"Continent {len(self.continent_names) + 1}"
                    size_sq_miles = area * 0.386102  # Assuming each pixel represents 0.386102 sq miles
                    center = self.find_continent_center(self.labels, label)
                    self.continent_names.append((name, size_sq_miles, center))

    def label_connected_regions(self, world_np):
        from scipy.ndimage import label
        threshold = 0.011  # This excludes deep water (below 0.01) and shallow water (0.01 - 0.03)
        binary_world = world_np > threshold
        labels, num_labels = label(binary_world)
        return labels, num_labels

    def find_continent_center(self, labels, label_value):
        rows, cols = np.where(labels == label_value)
        if len(rows) > 0 and len(cols) > 0:
            center_y = int(np.mean(rows))
            center_x = int(np.mean(cols))
            return center_x, center_y
        else:
            return None

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
                if event.key == pygame.K_SPACE:
                    self.generate_world()
            self.quit_button.handle_event(event)

    def update(self):
        pass

    def render(self):
        self.screen.fill((0, 0, 0))
        if self.world is not None:
            self.render_surface = self.world_surface.copy()
            self.screen.blit(self.render_surface, (0, 0))
        self.render_progress()
        self.render_continent_names()
        self.quit_button.draw(self.screen)
        pygame.display.flip()

    def render_progress(self):
        if self.progress < 100:
            font = pygame.font.Font(None, 36)
            text = font.render(f'Generating World: {self.progress:.1f}%', True, (255, 255, 255))
            self.screen.blit(text, (10, 10))

    def render_continent_names(self):
        font = pygame.font.Font(None, 24)
        text_color = (255, 255, 255)
        x = self.map_width - 200
        y = 20
        mouse_pos = pygame.mouse.get_pos()
        hovered_continent = None

        for name, size_sq_miles, center in self.continent_names:
            text = f"{name}: {size_sq_miles:.2f} sq miles"
            rendered_text = font.render(text, True, text_color)
            text_rect = rendered_text.get_rect(topleft=(x, y))
            self.screen.blit(rendered_text, text_rect)

            if text_rect.collidepoint(mouse_pos):
                hovered_continent = name

            y += 30

        if hovered_continent:
            self.highlight_continent(hovered_continent)

    def is_mouse_over_continent(self, mouse_pos, label_value):
        x, y = mouse_pos
        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            return self.labels[y, x] == label_value
        return False

    def highlight_continent(self, continent_name):
        label_value = None
        for name, size, center in self.continent_names:
            if name == continent_name:
                if center:
                    x, y = center
                    label_value = self.labels[y, x]
                break

        if label_value is not None:
            mask = self.labels == label_value
            highlight_surface = pygame.Surface((self.map_width, self.map_height), flags=pygame.SRCALPHA)
            highlight_surface.fill((0, 0, 0, 0))  # Make it completely transparent

            # Create a highlight color and blend it with the world colors
            highlight_color = pygame.Color(255, 255, 0, 128)  # Yellow color with 50% transparency
            highlight_surface.blit(self.world_surface, (0, 0))
            highlight_surface.set_alpha(128, pygame.RLEACCEL)  # Set the transparency

            # Apply the mask
            highlight = np.zeros((self.map_width, self.map_height, 4), dtype=np.uint8)
            highlight[mask] = highlight_color  # Only apply the highlight color where mask is True

            highlight_surface = pygame.surfarray.make_surface(highlight[:, :, :3])
            highlight_surface.set_alpha(128)  # Set the transparency

            self.screen.blit(highlight_surface, (0, 0))

    def update_world_surface(self):
        try:
            if self.world.shape != (self.map_width, self.map_height):
                raise ValueError("Generated world shape does not match surface dimensions")
            world_np = self.world.cpu().numpy()
            color_array = self.generate_color_array(world_np)
            pygame.surfarray.blit_array(self.world_surface, color_array)
        except Exception as e:
            print(f"Exception occurred: {e}")
            if hasattr(self, 'world'):
                print(f"self.world.shape: {self.world.shape}")
            raise

    def generate_color_array(self, world_np):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_tensor = torch.from_numpy(world_np).to(device)

        thresholds = torch.tensor([0.01, 0.02, 0.03, 0.1, 0.6, 0.7, 0.8], device=device)
        colors = torch.tensor([
            [0, 0, 128],     # Deep Water
            [0, 128, 255],   # Shallow Water
            [240, 230, 140], # Sand
            [34, 139, 34],   # Grass
            [60, 179, 113],  # Forest
            [160, 82, 45],   # Mountain
            [255, 250, 250], # Snow
            [128, 128, 128], # Rock
        ], dtype=torch.float32, device=device)

        color_array = torch.zeros((self.map_width, self.map_height, 3), dtype=torch.uint8, device=device)

        def blend_colors(color1, color2, factor):
            return (color1 * (1 - factor) + color2 * factor).to(dtype=torch.uint8)

        def smooth_elevation(elevations, kernel_size=3):
            padding = kernel_size // 2
            padded_elevations = torch.nn.functional.pad(elevations.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='replicate').squeeze()
            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) / (kernel_size * kernel_size)
            smoothed = torch.nn.functional.conv2d(padded_elevations.unsqueeze(0).unsqueeze(0), kernel).squeeze()
            return smoothed

        smoothed_tensor = smooth_elevation(world_tensor, 3)

        stylized_colors = torch.tensor([
            [0, 0, 64],
            [0, 64, 128],
            [200, 180, 100],
            [20, 100, 20],
            [40, 140, 80],
            [120, 60, 30],
            [220, 200, 200],
            [80, 80, 80],
        ], dtype=torch.float32, device=device)

        for i in range(len(thresholds) - 1):
            lower_threshold = thresholds[i]
            upper_threshold = thresholds[i + 1]
            mask = (smoothed_tensor >= lower_threshold) & (smoothed_tensor < upper_threshold)
            blend_factor = (smoothed_tensor[mask] - lower_threshold) / (upper_threshold - lower_threshold)
            color1 = colors[i]
            color2 = colors[i + 1]
            blended_colors = blend_colors(color1, color2, blend_factor.unsqueeze(-1))
            color_array[mask] = blended_colors

        mask = smoothed_tensor >= thresholds[-1]
        color_array[mask] = colors[-1].to(dtype=torch.uint8)

        color_array = color_array.cpu().numpy()
        return color_array

    def quit_game(self):
        self.running = False

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    game = Game(screen, 1280, 720)
    game.run()
    pygame.quit()
