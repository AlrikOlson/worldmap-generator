import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

class DeviceManager:
    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

class RandomManager:
    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

class ProgressCallback:
    @staticmethod
    def notify(progress_callback, percentage, message):
        if progress_callback:
            progress_callback(percentage, message)

class GaussianFilter:
    @staticmethod
    def apply(world, device, kernel_size=10, sigma=1.7):
        kernel = torch.tensor(GaussianFilter.gaussian_kernel(kernel_size, sigma), device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        world = world.unsqueeze(0).unsqueeze(0)
        world = torch.nn.functional.conv2d(world, kernel, padding=kernel_size // 2)
        return world.squeeze(0).squeeze(0)
    
    @staticmethod
    def gaussian_kernel(size, sigma):
        kx = torch.arange(size, dtype=torch.float32) - size // 2
        kx = torch.exp(-0.5 * (kx / sigma) ** 2)
        kernel = kx.unsqueeze(1) * kx.unsqueeze(0)
        kernel /= kernel.sum()
        return kernel

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

class PerlinNoiseGenerator:
    def __init__(self, width, height, scale, octaves, persistence, lacunarity, device):
        self.width = width
        self.height = height
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.device = device

    def generate_perlin_noise(self):
        shape = (self.width, self.height)
        noise = torch.zeros(shape, device=self.device)
        frequency = 1
        amplitude = 1
        max_value = 0

        for _ in range(self.octaves):
            x_offset = random.uniform(0, 1000)
            y_offset = random.uniform(0, 1000)
            x = (torch.linspace(0, self.width - 1, self.width, device=self.device) / self.scale * frequency) + x_offset
            y = (torch.linspace(0, self.height - 1, self.height, device=self.device) / self.scale * frequency) + y_offset

            x_grid, y_grid = torch.meshgrid(x, y)
            noise += amplitude * self.perlin(x_grid, y_grid)

            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity

        return noise / max_value

    def perlin(self, x, y):
        xi = x.floor().long()
        yi = y.floor().long()
        xf = x.frac()
        yf = y.frac()

        u = self.fade(xf)
        v = self.fade(yf)

        aa = self.gradient(xi, yi)
        ab = self.gradient(xi, yi + 1)
        ba = self.gradient(xi + 1, yi)
        bb = self.gradient(xi + 1, yi + 1)

        x1 = torch.lerp(aa, ba, u)
        x2 = torch.lerp(ab, bb, u)
        return torch.lerp(x1, x2, v)

    @staticmethod
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def gradient(x, y):
        random = torch.sin(x * 12.9898 + y * 78.233) * 43758.5453123
        return random.frac() * 2 - 1

class GeologicalProcessor:
    def __init__(self, width, height, device, num_plates=10):
        self.width = width
        self.height = height
        self.device = device
        self.num_plates = num_plates

    def apply_geological_features(self, world, progress_callback=None):
        ProgressCallback.notify(progress_callback, 50.0, "Adding eroded features")
        world = self.add_eroded_features(world, progress_callback)
        return world

    def add_eroded_features(self, world, progress_callback=None):
        erosion_passes = 10
        for i in range(erosion_passes):
            world = self.erode_world(world)
            ProgressCallback.notify(progress_callback, 50.0 + i * 1, f"Applying erosion pass {i + 1}")
        return world

    def erode_world(self, world):
        world_reshaped = world.unsqueeze(0).unsqueeze(0)
        kernel = torch.tensor([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], device=self.device, dtype=torch.float32)

        neighbors = torch.nn.functional.conv2d(world_reshaped, kernel.unsqueeze(0).unsqueeze(0), padding=1)
        neighbors = neighbors.squeeze(0).squeeze(0)
        world -= 0.01 * (neighbors > 0).float()

        return world

class ErosionProcessor:
    def __init__(self, width, height, device):
        self.width = width
        self.height = height
        self.device = device

    def apply_hydraulic_erosion(self, world, progress_callback=None):
        erosion_iterations = 50
        erosion_rate = 0.01
        deposition_rate = 0.01
        evaporation_rate = 0.1
        water_level = 0.1

        for i in range(erosion_iterations):
            water_flow = self.calculate_water_flow(world, water_level)
            velocity = self.calculate_velocity(water_flow)

            erosion = erosion_rate * velocity
            deposition = deposition_rate * velocity
            world = world - erosion + deposition

            water_level -= evaporation_rate * water_level

            ProgressCallback.notify(progress_callback, 60.0 + i * 2, f"Applying hydraulic erosion iteration {i + 1}")

        return world

    def calculate_water_flow(self, world, water_level):
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], device=self.device, dtype=torch.float32)

        water_flow = torch.nn.functional.conv2d(world.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1)
        water_flow = water_flow.squeeze(0).squeeze(0)
        water_flow *= water_level

        return water_flow

    @staticmethod
    def calculate_velocity(water_flow):
        velocity = torch.sqrt(water_flow[:-1, :-1] ** 2 + water_flow[:-1, 1:] ** 2)
        return torch.nn.functional.pad(velocity, (0, 1, 0, 1), mode='constant', value=0)

