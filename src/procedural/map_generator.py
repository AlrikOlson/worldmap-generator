import torch
import random
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

def visualize_steps(steps, step_names):
    num_steps = len(steps)
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 5))
    for i, (world, step_name) in enumerate(zip(steps, step_names)):
        world_np = world.cpu().numpy()
        ax = axes[i]
        ax.imshow(world_np, cmap='terrain')
        ax.set_title(step_name)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

class MapGenerator:
    def __init__(self, width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
        self.width = width
        self.height = height
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed if seed is not None else random.randint(0, 1000)
        
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.noise_generator = PerlinNoiseGenerator(width, height, scale, octaves, persistence, lacunarity, self.device)
        self.geological_processor = GeologicalProcessor(width, height, self.device, 3)
        self.erosion_processor = ErosionProcessor(width, height, self.device)

    def generate(self, progress_callback=None):
        self.seed = random.randint(0, 1000)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        steps = []
        step_names = []
        
        if progress_callback:
            progress_callback(5.0, "Generating initial Perlin noise")
        
        world = self.noise_generator.generate_perlin_noise()
        world = (world + 1) / 2  # Normalize the noise
        steps.append(world.clone())
        step_names.append('Initial Noise')
        
        if progress_callback:
            progress_callback(20.0, "Applying geological features")
        
        world = self.geological_processor.apply_geological_features(world, progress_callback)
        steps.append(world.clone())
        step_names.append('After Geological Features')
        
        if progress_callback:
            progress_callback(60.0, "Applying hydraulic erosion")
        
        world = self.erosion_processor.apply_hydraulic_erosion(world, progress_callback)
        steps.append(world.clone())
        step_names.append('After Hydraulic Erosion')
        
        if progress_callback:
            progress_callback(80.0, "Applying Gaussian filter")
        
        world_np = world.cpu().numpy()
        world_np = gaussian_filter(world_np, sigma=1.7)
        world = torch.tensor(world_np).to(self.device)
        steps.append(world.clone())
        step_names.append('After Gaussian Filter')
        
        # Final normalization
        world = (world - world.min()) / (world.max() - world.min())
        
        if progress_callback:
            progress_callback(90.0, "Final normalization and visualization")
        
        # visualize_steps(steps, step_names)
        
        if progress_callback:
            progress_callback(100.0, "Map generation complete")
        
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
    
    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def gradient(self, x, y):
        random = torch.sin(x * 12.9898 + y * 78.233) * 43758.5453123
        return random.frac() * 2 - 1

class GeologicalProcessor:
    def __init__(self, width, height, device, num_plates=10):
        self.width = width
        self.height = height
        self.device = device
        self.num_plates = num_plates

    def apply_geological_features(self, world, progress_callback=None):
        if progress_callback:
            progress_callback(25.0, "Detecting plate boundaries")

        boundaries = self.detect_plate_boundaries(world)
        
        if progress_callback:
            progress_callback(30.0, "Assigning plates")

        plates = self.assign_plates(boundaries)

        if progress_callback:
            progress_callback(35.0, "Simulating tectonic plate movements")

        world = self.simulate_tectonic_plates(world, plates, progress_callback)
        
        if progress_callback:
            progress_callback(50.0, "Adding eroded features")

        world = self.add_eroded_features(world, progress_callback)
        return world

    def detect_plate_boundaries(self, world):
        world_np = world.cpu().numpy()
        sobel_x = sobel(world_np, axis=0)
        sobel_y = sobel(world_np, axis=1)
        magnitude = np.hypot(sobel_x, sobel_y)
        boundaries = torch.tensor(magnitude > magnitude.mean(), dtype=torch.float32, device=self.device)
        return boundaries

    def assign_plates(self, boundaries):
        from scipy.ndimage import label
        boundaries_np = boundaries.cpu().numpy()
        labeled_boundaries, num_features = label(boundaries_np == 0)
        return torch.tensor(labeled_boundaries, dtype=torch.int32, device=self.device), num_features

    def simulate_tectonic_plates(self, world, plates, progress_callback=None):
        plate_ids, num_plates = plates
        velocities = torch.tensor([(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)) for _ in range(num_plates)], device=self.device)

        for step in range(2):
            for plate_id in range(1, num_plates + 1):
                velocity = velocities[plate_id - 1]
                plate_mask = (plate_ids == plate_id).float()
                world = self.tectonic_displacement(world, plate_mask, velocity)
            if progress_callback:
                progress_callback(35.0 + step * 1.5, f"Simulating tectonic movement step {step+1}")

        return world

    def tectonic_displacement(self, world, plate_mask, velocity):
        x = torch.linspace(0, self.width - 1, self.width, device=self.device)
        y = torch.linspace(0, self.height - 1, self.height, device=self.device)
        x_grid, y_grid = torch.meshgrid(x, y)
        
        displacement = torch.exp(-((x_grid - x_grid.mean()) ** 2 + (y_grid - y_grid.mean()) ** 2) / (2 * (self.width / 10) ** 2))
        displacement *= (velocity[0] + velocity[1])
        return world + plate_mask * displacement

    def add_eroded_features(self, world, progress_callback=None):
        erosion_passes = 10
        for i in range(erosion_passes):
            world = self.erode_world(world)
            if progress_callback:
                progress_callback(50.0 + i * 1, f"Applying erosion pass {i+1}")

        return world

    def erode_world(self, world):
        eroded_world = world.clone()
        world_reshaped = world.unsqueeze(0).unsqueeze(0)
        kernel = torch.tensor([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], device=self.device, dtype=torch.float32)

        neighbors = torch.nn.functional.conv2d(world_reshaped, kernel.view(1, 1, 3, 3), padding=1)
        neighbors = neighbors.squeeze(0).squeeze(0)
        eroded_world -= 0.01 * (neighbors > 0).float()

        return eroded_world

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
                progress_callback(60.0 + i * 2, f"Applying hydraulic erosion iteration {i+1}")

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
