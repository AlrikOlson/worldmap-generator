# src/procedural/map_generator.py
import torch
import numpy as np
from noise import pnoise2
import random

class MapGenerator:
    def __init__(self, width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None, device='cuda'):
        self.width = width
        self.height = height
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed if seed is not None else random.randint(0, 1000)
        self.device = device

    def generate(self, progress_callback=None):
        random.seed(self.seed)
        base = random.randint(0, 1000)
        
        if progress_callback:
            progress_callback(5.0)  # Initial progress update

        # Creating grid for coordinates in chunks to allow progress updates
        x_coords = torch.linspace(0, self.width-1, self.width, device=self.device) / self.scale
        y_coords = torch.linspace(0, self.height-1, self.height, device=self.device) / self.scale
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords)

        if progress_callback:
            progress_callback(10.0)  # Progress update after meshgrid creation

        # Generate Perlin noise using CPU (since pnoise2 is CPU-based)
        world = np.zeros((self.width, self.height), dtype=np.float32)
        for x in range(self.width):
            for y in range(self.height):
                world[x, y] = pnoise2(x_grid[x, y].item(), y_grid[x, y].item(), octaves=self.octaves, persistence=self.persistence, lacunarity=self.lacunarity, base=base)

            if progress_callback:
                progress_callback(10.0 + 40.0 * (x / self.width))  # Progress update during Perlin noise generation

        world = torch.tensor(world, device=self.device, dtype=torch.float32)
        world = (world + 1) / 2  # Normalizing the noise
        
        if progress_callback:
            progress_callback(50.0)  # Progress update after initial noise generation

        world = self.apply_geological_features(world, progress_callback)
        return world

    def regenerate(self, seed=None, progress_callback=None):
        self.seed = seed if seed is not None else random.randint(0, 1000)
        return self.generate(progress_callback)

    def apply_geological_features(self, world, progress_callback=None):
        world = self.add_eroded_features(world, progress_callback)
        world = self.add_rivers(world, progress_callback)
        return world

    def add_eroded_features(self, world, progress_callback=None):
        erosion_passes = 10
        for i in range(erosion_passes):
            world = self.erode_world(world)
            if progress_callback:
                progress_callback(50.0 + i / erosion_passes * 25.0)  # Assuming erosion is 25% of the work.
        return world

    def erode_world(self, world):
        eroded_world = world.clone()
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                neighbors = self.get_neighbors(world, x, y)
                min_neighbor = min(neighbors, key=lambda k: world[k[0], k[1]])
                if world[x, y] > world[min_neighbor[0], min_neighbor[1]]:
                    eroded_world[x, y] -= 0.01
                    eroded_world[min_neighbor[0], min_neighbor[1]] += 0.01
        return eroded_world

    def get_neighbors(self, world, x, y):
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < self.width - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < self.height - 1:
            neighbors.append((x, y + 1))
        return neighbors

    def add_rivers(self, world, progress_callback=None):
        num_rivers = 5
        river_starts = self.find_high_points(world, num_rivers)
        for i, (start_x, start_y) in enumerate(river_starts):
            self.carve_river(world, start_x, start_y)
            if progress_callback:
                progress_callback(75.0 + (i + 1) / num_rivers * 25.0)  # Rivers are the remaining 25%
        return world

    def find_high_points(self, world, num_points):
        high_points = []
        world_cpu = world.cpu().numpy()  # Move to CPU for numpy max calculations
        for _ in range(num_points):
            max_value = np.max(world_cpu)
            max_indices = np.where(world_cpu == max_value)
            if len(max_indices[0]) > 0:
                index = random.randint(0, len(max_indices[0]) - 1)
                x, y = max_indices[0][index], max_indices[1][index]
                high_points.append((x, y))
                world_cpu[x, y] = -1  # Mark the selected point as used
        return high_points

    def carve_river(self, world, x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        river_length = 200
        prev_direction = None

        for _ in range(river_length):
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                break

            world[x, y] = 0.0

            neighbors = self.get_neighbors(world, x, y)
            min_neighbor = min(neighbors, key=lambda k: world[k[0], k[1]])

            # Introduce some randomness and directionality to the river path
            if prev_direction is None or random.random() < 0.1:
                prev_direction = random.choice(directions)
            else:
                min_neighbor = min([n for n in neighbors if n[0] - x == prev_direction[0] and n[1] - y == prev_direction[1]],
                                   key=lambda k: world[k[0], k[1]], default=min_neighbor)

            x, y = min_neighbor