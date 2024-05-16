# src/procedural/map_generator.py
import numpy as np
from noise import pnoise2

class MapGenerator:
    def __init__(self, width, height, scale=100.0):
        self.width = width
        self.height = height
        self.scale = scale

    def generate(self):
        world = np.zeros((self.width, self.height))

        for x in range(self.width):
            for y in range(self.height):
                # pnoise2 returns values in the range [-1.0, 1.0], we need to normalize them to [0, 1]
                value = pnoise2(x / self.scale, y / self.scale)
                normalized_value = (value + 1) / 2  # Normalize to [0, 1]
                world[x][y] = normalized_value

        return world
