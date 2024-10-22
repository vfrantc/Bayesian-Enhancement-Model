

import torch
def calculate_noise_map(x):
    """
    args:
        x (Tensor) : B C H W
    return:
        noise_map (Tensor): return Noise Map of shape B C H W

    """

    def gradient(x):
        def sub_gradient(x):
            left_shift_x, right_shift_x, grad = torch.zeros_like(
                x), torch.zeros_like(x), torch.zeros_like(x)
            left_shift_x[:, :, 0:-1] = x[:, :, 1:]
            right_shift_x[:, :, 1:] = x[:, :, 0:-1]
            grad = 0.5 * (left_shift_x - right_shift_x)
            return grad

        return sub_gradient(x), sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)
    low_after_awb = x.exp()
    color_map = low_after_awb / (low_after_awb.sum(dim=1, keepdims=True) + 1e-4)
    dx, dy = gradient(color_map)
    noise_map = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]

    return noise_map