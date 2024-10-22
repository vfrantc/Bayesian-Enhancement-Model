import argparse
from glob import glob
import os
import sys
from natsort import natsorted
from PIL import Image
import numpy as np
class DummyFile(object):
    def write(self, x): pass
sys.stdout = DummyFile()
from basicsr.metrics import getUCIQE, getUIQM
from basicsr.metrics.uciqe_uiqm import UIQM

sys.stdout = sys.__stdout__

def get_average_UCIQE_and_UICM(img_dir):
    img_paths = natsorted( glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.bmp')) )
    total_uciqe = 0
    total_uiqm = 0
    uiqm_inst = UIQM()
    for img_path in img_paths:

        image_RGB = Image.open(img_path)
        total_uciqe += getUCIQE(np.array(image_RGB))

        image = Image.open(img_path)
        original_width, original_height = image.size
        new_width = 256
        new_height = int((new_width / original_width) * original_height)
        resized_image = image.resize((new_width, new_height))
        image_RGB = np.array(resized_image)

        total_uiqm += getUIQM(image_RGB)
    average_uciqe = total_uciqe / len(img_paths)
    average_uiqm = total_uiqm / len(img_paths)

    return average_uciqe, average_uiqm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average UCIQE and UIQM for a directory of images.")
    parser.add_argument('img_dir', type=str, help="Path to the directory containing the images.")

    args = parser.parse_args()

    average_uciqe, average_uiqm = get_average_UCIQE_and_UICM(args.img_dir)

    print(f"Average UCIQE: {average_uciqe:.4f}")
    print(f"Average UIQM: {average_uiqm:.4f}")
