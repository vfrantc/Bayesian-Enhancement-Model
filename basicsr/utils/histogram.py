import torch
import torch.nn.functional as F
import numpy as np

def compute_histograms(image, patch_size=8, bin_count=256):
    """
    Compute KDE-based histograms for each channel and each patch of the image, normalized to PDF using PyTorch.

    Parameters:
    - image: A numpy array of shape (H, W, C) with pixel values in [0, 1]
    - patch_size: Size of the patches (height and width)
    - bin_count: Number of bins for the KDE estimation

    Returns:
    - kde_histograms: A torch tensor of shape (C, H // patch_size, W // patch_size, bin_count)
    """
    # Convert image to torch tensor
    image = np.copy(image)
    image = torch.tensor(image, dtype=torch.float32)

    H, W, C = image.shape
    assert C == 3, "The image must have 3 channels (RGB)."

    # Padding the image if necessary
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image.permute(2, 0, 1), (0, pad_w, 0, pad_h), mode='reflect').permute(1, 2, 0)

    H, W, _ = image.shape

    # Split the image into patches and compute KDE for each patch
    patches = image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    num_patches_H, num_patches_W, _, _, _ = patches.shape

    # Initialize the KDE histogram tensor
    kde_histograms = torch.zeros((C, num_patches_H, num_patches_W, bin_count), dtype=torch.float32)

    # Define the bin edges for the KDE estimation
    bin_edges = torch.linspace(0, 1, bin_count, device=image.device)

    # Compute KDE for each channel
    for c in range(C):
        for i in range(num_patches_H):
            for j in range(num_patches_W):
                patch = patches[i, j, c, :, :].flatten()

                # Compute KDE using a Gaussian kernel
                kde_values = torch.exp(-0.5 * ((patch[:, None] - bin_edges[None, :]) ** 2) / 0.01)  # Bandwidth = 0.1
                kde_values = kde_values.mean(dim=0)

                # Normalize to form a probability density
                kde_values = kde_values + 1e-5
                kde_values /= kde_values.sum()

                # Store the KDE values in the tensor
                kde_histograms[c, i, j] = kde_values

    return kde_histograms

def compute_histogram_diff(image1, image2, patch_size=8, bin_count=32):
    """
    Compute the histogram difference of image1 and image2.

    Parameters:
    - image1: a numpy array of shape (H, W, C) with pixel values in [0, 1]
    - image2: a numpy array of shape (H, W, C) with pixel values in [0, 1]
    - patch_size: Size of the patches (height and width)
    - bin_count: Number of bins for the histogram

    Returns:
    - histogram difference: A numpy array of shape (bin_count*C, H // patch_size, W // patch_size)
    """
    hist1 = compute_histograms(image1, patch_size, bin_count)
    hist2 = compute_histograms(image2, patch_size, bin_count)
    hist_diff = hist1 - hist2
    hist_diff = hist_diff.transpose([3, 0, 1 ,2])
    hist_diff = hist_diff.reshape(-1, hist_diff.shape[-2], hist_diff.shape[-1])
    # hist_diff = hist_diff.transpose([1 ,2, 3, 0])
    # hist_diff = hist_diff.reshape(hist_diff.shape[0], hist_diff.shape[1], -1)

    return hist_diff



def gaussian_kernel(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):

    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)

    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=0)
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=0)

    y_true_hist /= y_true_hist.sum()
    y_pred_hist /= y_pred_hist.sum()

    hist_distance = torch.mean(torch.abs(y_true_hist - y_pred_hist))

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):

    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)

    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=0)
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=0)

    y_true_hist /= y_true_hist.sum()
    y_pred_hist /= y_pred_hist.sum()

    hist_distance = torch.mean(torch.abs(y_true_hist - y_pred_hist))

