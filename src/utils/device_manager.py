# src/utils/device_manager.py

import torch


class DeviceManager:
    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
