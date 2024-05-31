# src/core/continent_labeler.py

import numpy as np
import random

# Constants needed from game.py
LAND_SEA_THRESHOLD = 0.4
MIN_CONTINENT_SIZE = 5000
AREA_CONVERSION_FACTOR = 0.386102


class ContinentLabeler:
    def __init__(self, world):
        self.world = world

    def label_continents(self):
        world_np = self.world.data.cpu().numpy().astype(np.float32)
        labels, num_labels = self.label_connected_regions(world_np)
        self.world.update_labels(labels)

        potential_names = [
            "Atlantis", "Elysium", "Arcadia", "Erewhon", "Shangri-La",
            "Avalon", "El Dorado", "Valhalla", "Utopia", "Narnia"
        ]
        random.shuffle(potential_names)

        continent_names = []
        for label in range(1, num_labels + 1):
            area = np.sum(labels == label)
            if area > MIN_CONTINENT_SIZE:
                region_elevations = world_np[labels == label]
                if np.all(region_elevations >= LAND_SEA_THRESHOLD):
                    if len(continent_names) < len(potential_names):
                        name = potential_names[len(continent_names)]
                    else:
                        name = f"Continent {len(continent_names) + 1}"
                    size_sq_miles = area * AREA_CONVERSION_FACTOR
                    center = self.find_continent_center(labels, label)
                    continent_names.append((name, size_sq_miles, center))

        self.world.update_continent_names(continent_names)

    def label_connected_regions(self, world_np):
        from scipy.ndimage import label
        binary_world = world_np >= LAND_SEA_THRESHOLD
        labels, num_labels = label(binary_world)
        return labels, num_labels

    def find_continent_center(self, labels, label_value):
        rows, cols = np.where(labels == label_value)
        if len(rows) > 0 and len(cols) > 0:
            center_y = int(np.mean(rows))
            center_x = int(np.mean(cols))
            return center_x, center_y
        else:
            return None
