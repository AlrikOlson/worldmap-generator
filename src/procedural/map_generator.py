import random
from src.utils.device_manager import DeviceManager
from src.utils.random_manager import RandomManager
from src.utils.progress_callback import ProgressCallback
from src.utils.filter import GaussianFilter
from src.procedural.perlin_noise import PerlinNoiseGenerator
from src.procedural.erosion import ErosionProcessor
from src.procedural.geological import GeologicalProcessor

class MapGenerator:
    def __init__(self, width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
        self.width = width
        self.height = height
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed if seed is not None else random.randint(0, 1000)

        self.device = DeviceManager.get_device()
        self.noise_generator = PerlinNoiseGenerator(width, height, scale, octaves, persistence, lacunarity, self.device)
        self.geological_processor = GeologicalProcessor(width, height, self.device, 3)
        self.erosion_processor = ErosionProcessor(width, height, self.device)

    def generate(self, progress_callback=None):
        self.seed = random.randint(0, 1000)
        RandomManager.set_seed(self.seed)

        ProgressCallback.notify(progress_callback, 5.0, "Generating initial Perlin noise")
        world = self.noise_generator.generate_perlin_noise()
        world = (world + 1) / 2  # Normalize the noise

        ProgressCallback.notify(progress_callback, 20.0, "Applying geological features")
        world = self.geological_processor.apply_geological_features(world, progress_callback)

        ProgressCallback.notify(progress_callback, 60.0, "Applying hydraulic erosion")
        world = self.erosion_processor.apply_hydraulic_erosion(world, progress_callback)

        ProgressCallback.notify(progress_callback, 80.0, "Applying Gaussian filter")
        world = GaussianFilter.apply(world, self.device)

        world = world[:self.width, :self.height]

        world = (world - world.min()) / (world.max() - world.min())

        ProgressCallback.notify(progress_callback, 90.0, "Final normalization and visualization")
        ProgressCallback.notify(progress_callback, 100.0, "Map generation complete")

        return world
