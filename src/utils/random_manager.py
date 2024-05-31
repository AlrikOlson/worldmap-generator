# src/utils/random_manager.py

import random
import torch

class RandomManager:
    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
