import torch

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

            if progress_callback:
                progress_callback(60.0 + i * 2, f"Applying hydraulic erosion iteration {i + 1}")

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