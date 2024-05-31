# src/procedural/geological.py

import torch

class GeologicalProcessor:
    def __init__(self, width, height, device, num_plates=10):
        self.width = width
        self.height = height
        self.device = device
        self.num_plates = num_plates

    def apply_geological_features(self, world, progress_callback=None):
        if progress_callback:
            progress_callback(50.0, "Adding eroded features")
        world = self.add_eroded_features(world, progress_callback)
        return world

    def add_eroded_features(self, world, progress_callback=None):
        erosion_passes = 10
        for i in range(erosion_passes):
            world = self.erode_world(world)
            if progress_callback:
                progress_callback(50.0 + i * 1, f"Applying erosion pass {i + 1}")
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