# src/procedural/map_generator.py
import numpy as np
from noise import pnoise2
import random

class MapGenerator:
    def __init__(self, width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
        self.width = width
        self.height = height
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed if seed is not None else random.randint(0, 1000)  # Default to a random seed

    def generate(self):
        random.seed(self.seed)  # Set the seed for the random number generator
        base = random.randint(0, 1000)  # Generate a consistent base for pnoise2
        world = np.zeros((self.width, self.height))

        for x in range(self.width):
            for y in range(self.height):
                value = pnoise2(x / self.scale,
                                y / self.scale,
                                octaves=self.octaves,
                                persistence=self.persistence,
                                lacunarity=self.lacunarity,
                                base=base)
                normalized_value = (value + 1) / 2  # Normalize to [0, 1]
                world[x][y] = normalized_value

        # Add rivers
        self.add_rivers(world)
        
        return world

    def regenerate(self, seed=None):
        self.seed = seed if seed is not None else random.randint(0, 1000)
        return self.generate()

    def add_rivers(self, world):
        num_rivers = 5  # Number of rivers
        river_width = 3  # Width of the river in cells

        for _ in range(num_rivers):
            start_x, start_y = self.find_high_point(world)
            self.carve_river(world, start_x, start_y, river_width)

    def find_high_point(self, world):
        # Find a high point to start the river
        high_value = -1
        high_point = (0, 0)
        for _ in range(100):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if world[x][y] > high_value:
                high_value = world[x][y]
                high_point = (x, y)
        return high_point

    def carve_river(self, world, x, y, width):
        # Carve the river by lowering the elevation
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, right, up, down
        river_length = 100  # Length of the river

        for _ in range(river_length):
            for i in range(-width, width):
                for j in range(-width, width):
                    if 0 <= x + i < self.width and 0 <= y + j < self.height:
                        world[x + i][y + j] = 0.0  # Set elevation to water level

            # Move to the next cell in a random direction
            direction = random.choice(directions)
            x += direction[0]
            y += direction[1]

            # Ensure the river doesn't go out of bounds
            if not (0 <= x < self.width and 0 <= y < self.height):
                break

        # Optionally, add turbulence to the river path
        for _ in range(river_length // 10):
            x += random.randint(-1, 1)
            y += random.randint(-1, 1)
            if 0 <= x < self.width and 0 <= y < self.height:
                world[x][y] = 0.0
