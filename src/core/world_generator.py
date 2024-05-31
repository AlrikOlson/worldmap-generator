from src.procedural.map_generator import MapGenerator
import torch


class WorldGenerator:
    def __init__(self, width, height, scale, octaves):
        self.width = width
        self.height = height
        self.map_gen = MapGenerator(width, height, scale=scale, octaves=octaves)

    def generate(self, progress_callback):
        world_data = self.map_gen.generate(progress_callback).to(torch.float32)
        return world_data
