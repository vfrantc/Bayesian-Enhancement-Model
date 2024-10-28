import numpy as np
import cv2
import random

def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    image_255 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image_255.shape) == 2:
        clahe_image = clahe.apply(image_255)
    else:
        yuv_image = cv2.cvtColor(image_255, cv2.COLOR_BGR2YUV)
        yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])
        clahe_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    clahe_image = (clahe_image - clahe_image.min()) / (clahe_image.max() - clahe_image.min())

    return clahe_image

def adjust_color_temperature(image, temperature_factor):
    image_float = image.astype(np.float32)
    adjustment = np.array([1.0, 1.0, 1.0])
    adjustment = np.array([temperature_factor, 1.0, 1.0 / temperature_factor])
    img_adjusted = image_float * adjustment
    img_adjusted = np.clip(img_adjusted, 0, 1)
    return img_adjusted

def adjust_contrast(image, contrast_factor):
    img_float = image.astype(np.float32)
    img_contrast = contrast_factor * (img_float - 0.5) + 0.5
    img_contrast = np.clip(img_contrast, 0, 1)
    return img_contrast


def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image_255 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    corrected = cv2.LUT(image_255, table)
    return np.clip(corrected / 255.0, 0, 1)

def adjust_brightness(image, factor=1):
    image_float = image.astype(np.float32)
    image_float = image_float * factor
    image_float = np.clip(image_float, 0, 1)
    return image_float

def adjust_brightness_nonlinear(image, gamma):
    image_float = image.astype(np.float32)
    img_nonlinear = np.power(image_float, gamma)
    img_nonlinear = np.clip(img_nonlinear, 0, 1)
    return img_nonlinear


def add_label_noise(image_np,
                    tem_mean=1, tem_var=0.03,
                    bright_mean=1.15, bright_var=0.15,
                    contrast_mean=1.15, contrast_var=0.15):
    if tem_mean != 1 or tem_var != 0:
        temperature_factor = np.random.normal(tem_mean, tem_var)
        image_np = adjust_color_temperature(image_np, temperature_factor)
    if bright_mean != 1 or bright_var != 0:
        bright_factor = np.random.normal(bright_mean, bright_var)
        image_np = adjust_brightness(image_np, factor=bright_factor)
    if contrast_mean != 1 or contrast_var != 0:
        contrast_factor = np.random.normal(contrast_mean, contrast_var)
        image_np = adjust_contrast(image_np, contrast_factor)

    return image_np