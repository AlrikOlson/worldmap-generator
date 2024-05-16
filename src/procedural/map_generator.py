import torch
import random
import numpy as np
from scipy.ndimage import gaussian_filter


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
        self.noise_generator = PerlinNoiseGenerator(width, height, scale, octaves, persistence, lacunarity, self.device)
        self.geological_processor = GeologicalProcessor(width, height, self.device)
        self.erosion_processor = ErosionProcessor(width, height, self.device)

    def generate(self, progress_callback=None):
        self.seed = random.randint(0, 1000)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if progress_callback:
            progress_callback(5.0)
        
        world = self.noise_generator.generate_perlin_noise()
        world = (world + 1) / 2  # Normalize the noise
        #world = torch.tensor(gaussian_filter(world.cpu().numpy(), sigma=1.2)).to(self.device)
        
        if progress_callback:
            progress_callback(40.0)
        
        world = self.geological_processor.apply_geological_features(world, progress_callback)
        world = self.erosion_processor.apply_hydraulic_erosion(world, progress_callback)
        world = torch.tensor(gaussian_filter(world.cpu().numpy(), sigma=3)).to(self.device)
        
        return world

    def regenerate(self, seed=None, progress_callback=None):
        self.seed = seed if seed is not None else random.randint(0, 1000)
        return self.generate(progress_callback)


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
    
    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def gradient(self, x, y):
        random = torch.sin(x * 12.9898 + y * 78.233) * 43758.5453123
        return random.frac() * 2 - 1


class GeologicalProcessor:
    def __init__(self, width, height, device):
        self.width = width
        self.height = height
        self.device = device

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
        neighbors = neighbors.squeeze(0).squeeze(0)
        eroded_world -= 0.01 * (neighbors > 0).float()
        
        return eroded_world

    def add_rivers(self, world, progress_callback=None):
        num_rivers = 20
        river_starts = self.find_edge_points(world, num_rivers)

        for i, (start_x, start_y) in enumerate(river_starts):
            self.carve_river(world, start_x, start_y)
            if progress_callback:
                progress_callback(70.0 + (i + 1) / num_rivers * 30.0)  # Rivers are the remaining 30%
        return world

    def find_edge_points(self, world, num_points):
        edge_points = []
        world_cpu = world.cpu().numpy()

        for _ in range(num_points):
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                x = random.randint(0, self.width - 1)
                y = 0
            elif edge == 'bottom':
                x = random.randint(0, self.width - 1)
                y = self.height - 1
            elif edge == 'left':
                x = 0
                y = random.randint(0, self.height - 1)
            else:  # 'right'
                x = self.width - 1
                y = random.randint(0, self.height - 1)

            edge_points.append((x, y))

        return edge_points

    def carve_river(self, world, x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        river_length = 200
        prev_direction = None

        for _ in range(river_length):
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                break

            world[x, y] = 0.0

            # Calculate the slope in each direction
            slopes = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    slope = world[nx, ny] - world[x, y]
                    slopes.append((slope, (dx, dy)))

            # Choose the direction with the steepest downward slope
            max_slope, max_direction = max(slopes, key=lambda x: x[0], default=(0, (0, 0)))

            if prev_direction is None or random.random() < 0.1:
                prev_direction = max_direction
            else:
                # Filter directions to consider only the previous direction and the steepest direction
                valid_directions = [d for d in slopes if d[1] == prev_direction or d[1] == max_direction]
                _, prev_direction = max(valid_directions, key=lambda x: x[0], default=(0, (0, 0)))

            x += prev_direction[0]
            y += prev_direction[1]

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


class ErosionProcessor:
    def __init__(self, width, height, device):
        self.width = width
        self.height = height
        self.device = device

    def apply_hydraulic_erosion(self, world, progress_callback=None):
        erosion_iterations = 10
        erosion_rate = 0.01
        deposition_rate = 0.01
        evaporation_rate = 0.1
        water_level = 0.1

        for i in range(erosion_iterations):
            water_flow = self.calculate_water_flow(world, water_level)
            velocity = self.calculate_velocity(water_flow)

            erosion = erosion_rate * velocity
            deposition = deposition_rate * velocity
            world -= erosion
            world += deposition

            water_level -= evaporation_rate * water_level

            if progress_callback:
                progress_callback(70.0 + i / erosion_iterations * 30.0)

        return world

    def calculate_water_flow(self, world, water_level):
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
        velocity = torch.zeros_like(water_flow)
        velocity[:-1, :-1] = torch.sqrt(water_flow[:-1, :-1] ** 2 + water_flow[:-1, 1:] ** 2)
        velocity[-1, :] = velocity[-2, :]
        velocity[:, -1] = velocity[:, -2]
        return velocity
