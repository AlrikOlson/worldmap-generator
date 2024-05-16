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
        self.seed = seed if seed is not None else random.randint(0, 1000)

    def generate(self):
        random.seed(self.seed)
        base = random.randint(0, 1000)
        world = np.zeros((self.width, self.height))

        for x in range(self.width):
            for y in range(self.height):
                value = pnoise2(x / self.scale,
                                y / self.scale,
                                octaves=self.octaves,
                                persistence=self.persistence,
                                lacunarity=self.lacunarity,
                                base=base)
                normalized_value = (value + 1) / 2
                world[x][y] = normalized_value

        world = self.apply_geological_features(world)
        return world

    def regenerate(self, seed=None):
        self.seed = seed if seed is not None else random.randint(0, 1000)
        return self.generate()

    def apply_geological_features(self, world):
        world = self.add_eroded_features(world)
        world = self.add_rivers(world)
        return world

    def add_eroded_features(self, world):
        erosion_passes = 10
        for _ in range(erosion_passes):
            world = self.erode_world(world)
        return world

    def erode_world(self, world):
        for _ in range(10000):
            x, y = random.randint(1, self.width - 2), random.randint(1, self.height - 2)
            for _ in range(5):
                neighbors = self.get_neighbors(world, x, y)
                min_neighbor = min(neighbors, key=lambda k: world[k[0], k[1]])
                if world[x, y] > world[min_neighbor[0], min_neighbor[1]]:
                    world[x, y] -= 0.01
                    world[min_neighbor[0], min_neighbor[1]] += 0.01
                x, y = min_neighbor
        return world

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

    def add_rivers(self, world):
        num_rivers = 5
        for _ in range(num_rivers):
            start_x, start_y = self.find_high_point(world)
            self.carve_river(world, start_x, start_y)
        return world

    def find_high_point(self, world):
        high_value = -1
        high_point = (0, 0)
        for _ in range(100):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if world[x][y] > high_value:
                high_value = world[x][y]
                high_point = (x, y)
        return high_point

    def carve_river(self, world, x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        river_length = 200

        for _ in range(river_length):
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                break

            world[x, y] = 0.0

            neighbors = self.get_neighbors(world, x, y)
            min_neighbor = min(neighbors, key=lambda k: world[k[0], k[1]])
            x, y = min_neighbor
