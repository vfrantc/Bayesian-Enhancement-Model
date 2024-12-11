import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

def compute_rci(model, input_low_img, input_high_img, device, quaternion=False, save_variance_map=False,
                output_dir=None, variance_map_filename=None):
    """
    Compute the Reflectance Consistency Index (RCI) for a given pair of low-light and normal-light images.

    Parameters:
    - model: The neural network model.
    - input_low_img: Low-light image (PIL Image or torch.Tensor).
    - input_high_img: Normal-light image (PIL Image or torch.Tensor).
    - device: torch.device.
    - quaternion: Boolean indicating if the model outputs quaternions (default False).
    - save_variance_map: Boolean indicating whether to save the variance map using matplotlib (default False).
    - output_dir: Directory to save the variance map (required if save_variance_map is True).
    - variance_map_filename: Filename to save the variance map as (optional).

    Returns:
    - normalized_metric: The RCI metric (between 0 and 1).
    """
    # Ensure input images are tensors and on the device
    transform = transforms.ToTensor()

    if isinstance(input_low_img, Image.Image):
        input_low_tensor = transform(input_low_img).unsqueeze(0).to(device)
    elif isinstance(input_low_img, torch.Tensor):
        input_low_tensor = input_low_img.to(device)
        if input_low_tensor.dim() == 3:
            input_low_tensor = input_low_tensor.unsqueeze(0)
    else:
        raise ValueError("input_low_img must be a PIL Image or torch.Tensor")

    if isinstance(input_high_img, Image.Image):
        input_high_tensor = transform(input_high_img).unsqueeze(0).to(device)
    elif isinstance(input_high_img, torch.Tensor):
        input_high_tensor = input_high_img.to(device)
        if input_high_tensor.dim() == 3:
            input_high_tensor = input_high_tensor.unsqueeze(0)
    else:
        raise ValueError("input_high_img must be a PIL Image or torch.Tensor")

    # List to store reflectance maps
    Rs = []

    # Decompose the low-light image
    with torch.no_grad():
        if quaternion:
            # Model outputs quaternions
            outputs = model(input_low_tensor)
            if isinstance(outputs, tuple):
                hat_Q1, _ = outputs
            else:
                hat_Q1 = outputs
            # Extract imaginary parts (channels 1, 2, 3)
            R_low = hat_Q1[:, 1:, :, :]  # Shape: [1, 3, H, W]
        else:
            outputs = model(input_low_tensor)
            if isinstance(outputs, tuple):
                R_low, _ = outputs
            else:
                R_low = outputs
        R_low_np = R_low.cpu().squeeze(0).numpy()  # Shape: (3, H, W)
        R_low_np = np.transpose(R_low_np, (1, 2, 0))  # Shape: (H, W, 3)
        Rs.append(R_low_np)

    # Generate blended images and decompose them
    num_steps = 10
    for i in range(1, num_steps):
        alpha = i / num_steps
        blended_tensor = (1 - alpha) * input_low_tensor + alpha * input_high_tensor
        with torch.no_grad():
            if quaternion:
                outputs = model(blended_tensor)
                if isinstance(outputs, tuple):
                    hat_Q1, _ = outputs
                else:
                    hat_Q1 = outputs
                R_blended = hat_Q1[:, 1:, :, :]
            else:
                outputs = model(blended_tensor)
                if isinstance(outputs, tuple):
                    R_blended, _ = outputs
                else:
                    R_blended = outputs
            R_blended_np = R_blended.cpu().squeeze(0).numpy()
            R_blended_np = np.transpose(R_blended_np, (1, 2, 0))
            Rs.append(R_blended_np)

    # Decompose the high-light image
    with torch.no_grad():
        if quaternion:
            outputs = model(input_high_tensor)
            if isinstance(outputs, tuple):
                hat_Q1, _ = outputs
            else:
                hat_Q1 = outputs
            R_high = hat_Q1[:, 1:, :, :]
        else:
            outputs = model(input_high_tensor)
            if isinstance(outputs, tuple):
                R_high, _ = outputs
            else:
                R_high = outputs
        R_high_np = R_high.cpu().squeeze(0).numpy()
        R_high_np = np.transpose(R_high_np, (1, 2, 0))
        Rs.append(R_high_np)

    # Stack reflectance maps and compute variance
    Rs_array = np.stack(Rs, axis=0)  # Shape: (num_images, H, W, 3)
    variance_per_pixel = np.var(Rs_array, axis=0)  # Shape: (H, W, 3)
    max_variance = np.max(variance_per_pixel)
    normalized_metric = 1 - (max_variance / 0.25)
    normalized_metric = np.clip(normalized_metric, 0, 1)

    # Optionally save the variance map
    if save_variance_map:
        if output_dir is None:
            raise ValueError("output_dir must be specified when save_variance_map is True")
        variance_max = np.max(variance_per_pixel, axis=2)  # Shape: (H, W)
        plt.imshow(variance_max, cmap='hot')
        plt.axis('off')
        plt.colorbar()
        os.makedirs(output_dir, exist_ok=True)
        if variance_map_filename is None:
            variance_map_filename = 'variance_map.png'
        variance_map_path = os.path.join(output_dir, variance_map_filename)
        plt.savefig(variance_map_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    return normalized_metric
