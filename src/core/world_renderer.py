# src/core/world_renderer.py

import pygame
import torch
import numpy as np
from src.utils.device_manager import DeviceManager

# Constants needed from game.py
LAND_SEA_THRESHOLD = 0.4
KERNEL_SIZE = 5
SHALLOW_DEEP_BLEND_FACTOR = 35.0
WORLD_REGION_COLORS = [
    [0, 34, 102],  # DEEP_WATER_COLOR
    [51, 153, 204],  # SHALLOW_WATER_COLOR
    [210, 184, 159],  # BEACH_COLOR
    [34, 139, 34],  # GRASSLAND_COLOR
    [85, 107, 47],  # FOREST_COLOR
    [169, 169, 169],  # MOUNTAIN_COLOR
    [255, 255, 255]  # SNOW_COLOR
]
HIGHLIGHT_RGBA = (255, 255, 0, 128)
HIGHLIGHT_ALPHA = 128


class WorldRenderer:
    def __init__(self, world, screen):
        self.world = world
        self.screen = screen
        self.world_surface = pygame.Surface((world.width, world.height))
        self.render_surface = self.world_surface.copy()

    def update_world_surface(self):
        try:
            if self.world.data.shape[:2] != (self.world.width, self.world.height):
                raise ValueError("Generated world shape does not match surface dimensions")
            world_np = self.world.data.cpu().numpy().astype(np.float32)  # Ensure data is float32
            color_array = self.generate_color_array(world_np)
            pygame.surfarray.blit_array(self.world_surface, color_array)
        except Exception as e:
            print(f"Exception occurred: {e}")
            if hasattr(self.world, 'data'):
                print(f"self.world.data.shape: {self.world.data.shape}")
            raise

    def generate_color_array(self, world_np):
        device = DeviceManager.get_device()
        world_tensor = torch.from_numpy(world_np).to(device).to(torch.float32)

        land_mask = world_tensor >= LAND_SEA_THRESHOLD
        land_elevations = world_tensor[land_mask]

        deep_water_threshold = np.percentile(world_np[world_np < LAND_SEA_THRESHOLD], 0.0)
        shallow_water_threshold = np.percentile(world_np[world_np < LAND_SEA_THRESHOLD], 90.0)
        grassland_threshold = np.percentile(land_elevations.cpu().numpy(), 25.0)
        forest_threshold = np.percentile(land_elevations.cpu().numpy(), 50.0)
        mountain_threshold = np.percentile(land_elevations.cpu().numpy(), 99.0)
        snow_threshold = np.percentile(land_elevations.cpu().numpy(), 99.9)

        thresholds = torch.tensor([
            deep_water_threshold,
            shallow_water_threshold,
            LAND_SEA_THRESHOLD,
            grassland_threshold,
            forest_threshold,
            mountain_threshold,
            snow_threshold
        ], device=device, dtype=torch.float32)

        colors = torch.tensor(WORLD_REGION_COLORS, dtype=torch.float32, device=device)

        color_array = torch.zeros((self.world.width, self.world.height, 3), dtype=torch.uint8, device=device)

        def blend_colors(color1, color2, factor):
            return (color1 * (1 - factor) + color2 * factor).to(dtype=torch.uint8)

        def smooth_elevation(elevations, kernel_size=KERNEL_SIZE):
            padding = kernel_size // 2
            padded_elevations = torch.nn.functional.pad(elevations.unsqueeze(0).unsqueeze(0),
                                                        (padding, padding, padding, padding),
                                                        mode='replicate').squeeze()
            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) / (kernel_size * kernel_size)
            smoothed = torch.nn.functional.conv2d(padded_elevations.unsqueeze(0).unsqueeze(0), kernel).squeeze()
            return smoothed

        smoothed_tensor = smooth_elevation(world_tensor.to(torch.float32), KERNEL_SIZE)

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
        self.screen.fill((0, 0, 0))
        if self.world.data is not None:
            self.render_surface = self.world_surface.copy()
            self.screen.blit(self.render_surface, (0, 0))
        self.render_continent_names()

    def render_continent_names(self):
        font = pygame.font.Font(None, 24)
        x = self.world.width - 200
        y = 20
        mouse_pos = pygame.mouse.get_pos()
        hovered_continent = None

        for name, size_sq_miles, center in self.world.continent_names:
            text = f"{name}: {size_sq_miles:.2f} sq miles"
            rendered_text = font.render(text, True, (255, 255, 255))
            text_rect = rendered_text.get_rect(topleft=(x, y))
            self.screen.blit(rendered_text, text_rect)

            if text_rect.collidepoint(mouse_pos):
                hovered_continent = name

            y += 30

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
