import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class PairDataset(Dataset):
    def __init__(self, input_dir, gt_dir, patch_size):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.image_names = os.listdir(self.input_dir)
        self.image_names = [
            img for img in self.image_names
            if img.endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        input_name = self.image_names[idx]
        gt_name = input_name  # Assuming the same name for GT image
        input_path = os.path.join(self.input_dir, input_name)
        gt_path = os.path.join(self.gt_dir, gt_name)

        # Load images and convert to RGB in range 0..1
        input_image = Image.open(input_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')
        input_image = np.array(input_image, dtype='float32') / 255.0
        gt_image = np.array(gt_image, dtype='float32') / 255.0

        h, w, _ = input_image.shape
        if h > self.patch_size and w > self.patch_size:
            # Randomly select top-left corner for cropping
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)

            # Crop both input and gt images
            input_image = input_image[top: top + self.patch_size, left: left + self.patch_size, :].copy()
            gt_image = gt_image[top: top + self.patch_size, left: left + self.patch_size, :].copy()

        # Data augmentation: random horizontal and vertical flips
        if np.random.rand() < 0.5:
            input_image = np.flipud(input_image).copy()
            gt_image = np.flipud(gt_image).copy()
        if np.random.rand() < 0.5:
            input_image = np.fliplr(input_image).copy()
            gt_image = np.fliplr(gt_image).copy()

        # Data augmentation: random rotation
        if np.random.rand() < 0.5:
            rotation_times = np.random.randint(1, 4)
            input_image = np.rot90(input_image, rotation_times).copy()
            gt_image = np.rot90(gt_image, rotation_times).copy()

        # Convert images to PyTorch tensors and rearrange dimensions to (C, H, W)
        input_image = torch.from_numpy(input_image.transpose((2, 0, 1))).float()
        gt_image = torch.from_numpy(gt_image.transpose((2, 0, 1))).float()

        return input_image, gt_image
