import pygame
import threading
import numpy as np
import torch
from procedural.map_generator import MapGenerator
from ui.button import Button
import random
from scipy.ndimage import measurements

# Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 10
QUIT_TEXT_SIZE = 36
CONTINENT_TEXT_SIZE = 24
TEXT_X_OFFSET = 200
START_Y_OFFSET = 20
CONTINENT_TEXT_Y_PADDING = 30
PROGRESS_TEXT_X = 10
PROGRESS_TEXT_Y = 10
BACKGROUND_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
QUIT_BUTTON_COLOR = (150, 0, 0)
QUIT_BUTTON_HOVER_COLOR = (200, 0, 0)
HIGHLIGHT_ALPHA = 128
HIGHLIGHT_RGBA = (255, 255, 0, HIGHLIGHT_ALPHA)
MAP_SCALE = 400.0
MAP_OCTAVES = 48
MIN_CONTINENT_SIZE = 5000
AREA_CONVERSION_FACTOR = 0.386102
KERNEL_SIZE = 3
FPS = 60

SHALLOW_DEEP_BLEND_FACTOR = 25.0  # You can adjust this value to control the smoothness

# Threshold for land/sea
LAND_SEA_THRESHOLD = 0.7

# Absolute thresholds for water
DEEP_WATER_THRESHOLD = 0.0
SHALLOW_WATER_THRESHOLD = 0.65  # Adjust as needed

# Percentiles for land regions
GRASSLAND_PERCENTILE = 0.2
FOREST_PERCENTILE = 0.6
MOUNTAIN_PERCENTILE = 0.9
SNOW_PERCENTILE = 0.99

# Realistic colors
DEEP_WATER_COLOR = [0, 34, 102]  # Dark Blue
SHALLOW_WATER_COLOR = [102, 178, 255]  # Light Blue
BEACH_COLOR = [255, 229, 204]  # Sandy Yellow
GRASSLAND_COLOR = [124, 252, 0]  # Light Green
FOREST_COLOR = [34, 139, 34]  # Dark Green
MOUNTAIN_COLOR = [169, 169, 169]  # Gray
SNOW_COLOR = [255, 255, 255]  # White

WORLD_REGION_COLORS = [
    DEEP_WATER_COLOR,
    SHALLOW_WATER_COLOR,
    BEACH_COLOR,
    GRASSLAND_COLOR,
    FOREST_COLOR,
    MOUNTAIN_COLOR,
    SNOW_COLOR
]

class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = None
        self.labels = None
        self.continent_names = []

    def update_data(self, data):
        self.data = data

    def update_labels(self, labels):
        self.labels = labels

    def update_continent_names(self, names):
        self.continent_names = names

class WorldGenerator:
    def __init__(self, width, height, scale, octaves):
        self.width = width
        self.height = height
        self.map_gen = MapGenerator(width, height, scale=scale, octaves=octaves)

    def generate(self, progress_callback):
        world_data = self.map_gen.generate(progress_callback)
        return world_data

class ContinentLabeler:
    def __init__(self, world):
        self.world = world

    def label_continents(self):
        world_np = self.world.data.cpu().numpy()
        labels, num_labels = self.label_connected_regions(world_np)
        self.world.update_labels(labels)

        potential_names = [
            "Atlantis", "Elysium", "Arcadia", "Erewhon", "Shangri-La",
            "Avalon", "El Dorado", "Valhalla", "Utopia", "Narnia"
        ]
        random.shuffle(potential_names)

        continent_names = []
        for label in range(1, num_labels + 1):
            area = np.sum(labels == label)
            if area > MIN_CONTINENT_SIZE:
                region_elevations = world_np[labels == label]
                if np.all(region_elevations >= LAND_SEA_THRESHOLD):
                    if len(continent_names) < len(potential_names):
                        name = potential_names[len(continent_names)]
                    else:
                        name = f"Continent {len(continent_names) + 1}"
                    size_sq_miles = area * AREA_CONVERSION_FACTOR
                    center = self.find_continent_center(labels, label)
                    continent_names.append((name, size_sq_miles, center))

        self.world.update_continent_names(continent_names)

    def label_connected_regions(self, world_np):
        from scipy.ndimage import label
        binary_world = world_np >= LAND_SEA_THRESHOLD
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

class WorldRenderer:
    def __init__(self, world, screen):
        self.world = world
        self.screen = screen
        self.world_surface = pygame.Surface((world.width, world.height))
        self.render_surface = self.world_surface.copy()

    def update_world_surface(self):
        try:
            if self.world.data.shape != (self.world.width, self.world.height):
                raise ValueError("Generated world shape does not match surface dimensions")
            world_np = self.world.data.cpu().numpy()
            color_array = self.generate_color_array(world_np)
            pygame.surfarray.blit_array(self.world_surface, color_array)
        except Exception as e:
            print(f"Exception occurred: {e}")
            if hasattr(self.world, 'data'):
                print(f"self.world.data.shape: {self.world.data.shape}")
            raise

    def generate_color_array(self, world_np):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_tensor = torch.from_numpy(world_np).to(device)

        land_mask = world_tensor >= LAND_SEA_THRESHOLD
        land_elevations = world_tensor[land_mask]

        grassland_threshold = np.percentile(land_elevations.cpu().numpy(), GRASSLAND_PERCENTILE * 100)
        forest_threshold = np.percentile(land_elevations.cpu().numpy(), FOREST_PERCENTILE * 100)
        mountain_threshold = np.percentile(land_elevations.cpu().numpy(), MOUNTAIN_PERCENTILE * 100)
        snow_threshold = np.percentile(land_elevations.cpu().numpy(), SNOW_PERCENTILE * 100)

        thresholds = torch.tensor([
            DEEP_WATER_THRESHOLD,
            SHALLOW_WATER_THRESHOLD,
            LAND_SEA_THRESHOLD,
            grassland_threshold,
            forest_threshold,
            mountain_threshold,
            snow_threshold
        ], device=device)

        colors = torch.tensor(WORLD_REGION_COLORS, dtype=torch.float32, device=device)

        color_array = torch.zeros((self.world.width, self.world.height, 3), dtype=torch.uint8, device=device)

        def blend_colors(color1, color2, factor):
            return (color1 * (1 - factor) + color2 * factor).to(dtype=torch.uint8)

        def smooth_elevation(elevations, kernel_size=KERNEL_SIZE):
            padding = kernel_size // 2
            padded_elevations = torch.nn.functional.pad(elevations.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='replicate').squeeze()
            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) / (kernel_size * kernel_size)
            smoothed = torch.nn.functional.conv2d(padded_elevations.unsqueeze(0).unsqueeze(0), kernel).squeeze()
            return smoothed

        smoothed_tensor = smooth_elevation(world_tensor, KERNEL_SIZE)

        for i in range(len(thresholds) - 1):
            lower_threshold = thresholds[i]
            upper_threshold = thresholds[i + 1]
            mask = (smoothed_tensor >= lower_threshold) & (smoothed_tensor < upper_threshold)

            blend_factor = (smoothed_tensor[mask] - lower_threshold) / (upper_threshold - lower_threshold)

            if i == 0:  # Deep to shallow water transition
                blend_factor = torch.pow(blend_factor, SHALLOW_DEEP_BLEND_FACTOR)

            color1 = colors[i]
            color2 = colors[i + 1]
            blended_colors = blend_colors(color1, color2, blend_factor.unsqueeze(-1))
            color_array[mask] = blended_colors

        mask = smoothed_tensor >= thresholds[-1]
        color_array[mask] = colors[-1].to(dtype=torch.uint8)

        color_array = color_array.cpu().numpy()
        return color_array
    
    def render(self):
        self.screen.fill(BACKGROUND_COLOR)
        if self.world.data is not None:
            self.render_surface = self.world_surface.copy()
            self.screen.blit(self.render_surface, (0, 0))
        self.render_continent_names()

    def render_continent_names(self):
        font = pygame.font.Font(None, CONTINENT_TEXT_SIZE)
        x = self.world.width - TEXT_X_OFFSET
        y = START_Y_OFFSET
        mouse_pos = pygame.mouse.get_pos()
        hovered_continent = None

        for name, size_sq_miles, center in self.world.continent_names:
            text = f"{name}: {size_sq_miles:.2f} sq miles"
            rendered_text = font.render(text, True, TEXT_COLOR)
            text_rect = rendered_text.get_rect(topleft=(x, y))
            self.screen.blit(rendered_text, text_rect)

            if text_rect.collidepoint(mouse_pos):
                hovered_continent = name

            y += CONTINENT_TEXT_Y_PADDING

        if hovered_continent:
            self.highlight_continent(hovered_continent)

    def is_mouse_over_continent(self, mouse_pos, label_value):
        x, y = mouse_pos
        if 0 <= x < self.world.width and 0 <= y < self.world.height:
            return self.world.labels[y, x] == label_value
        return False

    def highlight_continent(self, continent_name):
        label_value = None
        for name, size, center in self.world.continent_names:
            if name == continent_name:
                if center:
                    x, y = center
                    label_value = self.world.labels[y, x]
                break

        if label_value is not None:
            mask = self.world.labels == label_value
            highlight_surface = pygame.Surface((self.world.width, self.world.height), flags=pygame.SRCALPHA)
            highlight_surface.fill((0, 0, 0, 0))

            highlight_color = pygame.Color(*HIGHLIGHT_RGBA)
            highlight_surface.blit(self.world_surface, (0, 0))
            highlight_surface.set_alpha(HIGHLIGHT_ALPHA, pygame.RLEACCEL)

            highlight = np.zeros((self.world.width, self.world.height, 4), dtype=np.uint8)
            highlight[mask] = highlight_color

            highlight_surface = pygame.surfarray.make_surface(highlight[:, :, :3])
            highlight_surface.set_alpha(HIGHLIGHT_ALPHA)

            self.screen.blit(highlight_surface, (0, 0))

class Game:
    def __init__(self, screen, map_width, map_height):
        self.screen = screen
        self.map_width = map_width
        self.map_height = map_height
        self.clock = pygame.time.Clock()
        self.running = True
        self.world = World(map_width, map_height)
        self.world_generator = WorldGenerator(map_width, map_height, scale=MAP_SCALE, octaves=MAP_OCTAVES)
        self.continent_labeler = ContinentLabeler(self.world)
        self.world_renderer = WorldRenderer(self.world, self.screen)
        self.progress = 0
        self.generate_world()

        self.quit_button = Button(
            rect=(map_width - BUTTON_WIDTH - BUTTON_MARGIN, map_height - BUTTON_HEIGHT - BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT),
            color=QUIT_BUTTON_COLOR,
            hover_color=QUIT_BUTTON_HOVER_COLOR,
            text='Quit',
            font=pygame.font.Font(None, QUIT_TEXT_SIZE),
            text_color=TEXT_COLOR,
            callback=self.quit_game
        )

    def generate_world(self):
        self.progress = 0
        threading.Thread(target=self.run_map_generation).start()

    def run_map_generation(self):
        world_data = self.world_generator.generate(self.update_progress)
        self.world.update_data(world_data)
        self.continent_labeler.label_continents()
        self.world_renderer.update_world_surface()

    def update_progress(self, progress):
        self.progress = progress

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)

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
        self.world_renderer.render()
        self.render_progress()
        self.quit_button.draw(self.screen)
        pygame.display.flip()

    def render_progress(self):
        if self.progress < 100:
            font = pygame.font.Font(None, QUIT_TEXT_SIZE)
            text = font.render(f'Generating World: {self.progress:.1f}%', True, TEXT_COLOR)
            self.screen.blit(text, (PROGRESS_TEXT_X, PROGRESS_TEXT_Y))

    def quit_game(self):
        self.running = False

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    game = Game(screen, WINDOW_WIDTH, WINDOW_HEIGHT)
    game.run()
    pygame.quit()
