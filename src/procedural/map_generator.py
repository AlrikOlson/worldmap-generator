import torch
import random
import numpy as np
from scipy.ndimage import gaussian_filter  # Add this import

class MapGenerator:
    def __init__(self, width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None, device='cuda'):
        self.width = width
        self.height = height
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed if seed is not None else random.randint(0, 1000)
        self.device = torch.device(device)

    def generate(self, progress_callback=None):
        self.seed = random.randint(0, 1000)  # Generate a new random seed each time
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if progress_callback:
            progress_callback(5.0)  # Initial progress update

        # Using GPU-accelerated noise generation with PyTorch
        world = self.generate_perlin_noise(self.width, self.height, self.scale, self.octaves, self.persistence, self.lacunarity)
        world = (world + 1) / 2  # Normalizing the noise
        
        # Apply Gaussian Blur for smoothing
        world = torch.tensor(gaussian_filter(world.cpu().numpy(), sigma=3.5)).to(self.device)  # Move back to GPU

        if progress_callback:
            progress_callback(40.0)  # Progress update after initial noise generation

        world = self.apply_geological_features(world, progress_callback)
        # Apply hydraulic erosion simulation
        world = self.apply_hydraulic_erosion(world, progress_callback)
        return world

    def generate_perlin_noise(self, width, height, scale, octaves, persistence, lacunarity):
        shape = (width, height)
        noise = torch.zeros(shape, device=self.device)
        
        frequency = 1
        amplitude = 1
        max_value = 0
        
        for _ in range(octaves):
            # Incorporate the seed into the offsets
            x_offset = random.uniform(0, 1000)
            y_offset = random.uniform(0, 1000)
            x = (torch.linspace(0, width - 1, width, device=self.device) / scale * frequency) + x_offset
            y = (torch.linspace(0, height - 1, height, device=self.device) / scale * frequency) + y_offset

            x_grid, y_grid = torch.meshgrid(x, y)
            noise += amplitude * self.perlin(x_grid, y_grid)

            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        return noise / max_value

    def perlin(self, x, y):
        # Implementation of 2D Perlin noise on GPU
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
    
    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def gradient(self, x, y):
        random = torch.sin(x * 12.9898 + y * 78.233) * 43758.5453
        return random.frac() * 2 - 1

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
                progress_callback(40 + i / erosion_passes * 30.0)  # Assuming erosion is 30% of the work.
        return world

    def erode_world(self, world):
        eroded_world = world.clone()
        world_reshaped = world.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, height, width]
        kernel = torch.tensor([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], device=self.device, dtype=torch.float32)
        
        neighbors = torch.nn.functional.conv2d(world_reshaped, kernel.view(1, 1, 3, 3), padding=1)
        neighbors = neighbors.squeeze(0).squeeze(0)  # Shape: [height, width]
        eroded_world -= 0.01 * (neighbors > 0).float()
        
        return eroded_world

    def add_rivers(self, world, progress_callback=None):
        num_rivers = 5
        river_starts = self.find_high_points(world, num_rivers)

        for i, (start_x, start_y) in enumerate(river_starts):
            self.carve_river(world, start_x, start_y)
            if progress_callback:
                progress_callback(70.0 + (i + 1) / num_rivers * 30.0)  # Rivers are the remaining 30%
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

    
    def apply_hydraulic_erosion(self, world, progress_callback=None):
        # Implement hydraulic erosion simulation on the GPU
        erosion_iterations = 10
        erosion_rate = 0.01
        deposition_rate = 0.01
        evaporation_rate = 0.1
        water_level = 0.1

        for i in range(erosion_iterations):
            # Calculate water flow and velocity using finite differences
            water_flow = self.calculate_water_flow(world, water_level)
            velocity = self.calculate_velocity(water_flow)

            # Update the terrain height based on erosion and deposition
            erosion = erosion_rate * velocity
            deposition = deposition_rate * velocity
            world -= erosion
            world += deposition

            # Update the water level based on evaporation
            water_level -= evaporation_rate * water_level

            if progress_callback:
                progress_callback(70.0 + i / erosion_iterations * 30.0)  # Assuming hydraulic erosion is the last 30% of the work

        return world

    def calculate_water_flow(self, world, water_level):
        # Calculate water flow using finite differences
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], device=self.device, dtype=torch.float32)

        water_flow = torch.nn.functional.conv2d(world.unsqueeze(0).unsqueeze(0), kernel.view(1, 1, 3, 3), padding=1)
        water_flow = water_flow.squeeze(0).squeeze(0)
        water_flow *= water_level

        return water_flow

    def calculate_velocity(self, water_flow):
        # Calculate velocity based on water flow
        velocity = torch.zeros_like(water_flow)
        velocity[:-1, :] = torch.sqrt(water_flow[:-1, :] ** 2 + water_flow[:-1, :-1] ** 2)
        velocity[-1, :] = velocity[-2, :]  # Copy the last row from the previous row
        velocity[:, -1] = velocity[:, -2]  # Copy the last column from the previous column
        return velocity