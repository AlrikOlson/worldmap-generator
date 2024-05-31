import torch
import random

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

            x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
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