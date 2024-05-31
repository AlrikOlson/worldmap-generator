import torch

class GaussianFilter:
    @staticmethod
    def apply(world, device, kernel_size=10, sigma=1.7):
        kernel = torch.from_numpy(GaussianFilter.gaussian_kernel(kernel_size, sigma)).to(device=device,
                                                                                         dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        world = world.unsqueeze(0).unsqueeze(0)
        world = torch.nn.functional.conv2d(world, kernel, padding=kernel_size // 2)
        return world.squeeze(0).squeeze(0)

    @staticmethod
    def gaussian_kernel(size, sigma):
        kx = torch.arange(size, dtype=torch.float32) - size // 2
        kx = torch.exp(-0.5 * (kx / sigma) ** 2)
        kernel = kx.unsqueeze(1) * kx.unsqueeze(0)
        kernel /= kernel.sum()
        return kernel.numpy()
