import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import argparse
import numpy as np

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

from quaternion import hamilton_product
from rci import compute_rci
# from models import Model2, Model3, Model4  # Add these if needed

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate the model on test images and compute metrics.')
    parser.add_argument('--model_checkpoint', type=str, default='./checkpoint/model_decomp_wavelet_300.pth', help='Path to the model checkpoint file.')
    parser.add_argument('--test_input_dir', type=str, default='./data/test/LOL/input', help='Path to the test input images directory.')
    parser.add_argument('--test_gt_dir', type=str, default='./data/test/LOL/gt', help='Path to the ground truth images directory.')
    parser.add_argument('--output_dir', type=str, default='./evaluation', help='Path to save evaluation results.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the evaluation on (cuda or cpu).')
    parser.add_argument('--compute_rci', default=True, action='store_true', help='Whether to compute RCI metric.')
    parser.add_argument('--model_type', type=str, choices=['model1','model2','model3','model4'], default='model1',
                        help='Which model architecture was used.')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load the selected model type
    if args.model_type == 'model1':
        from model1 import Decomp
        model = Decomp().to(device)
    elif args.model_type == 'model2':
        from model2 import Decomp
        model = Decomp().to(device)  # Placeholder, replace with actual model2
    elif args.model_type == 'model3':
        from model3 import Decomp
        model = Decomp().to(device)  # Placeholder, replace with actual model3
    elif args.model_type == 'model4':
        from model4 import Decomp
        model = Decomp().to(device)  # Placeholder, replace with actual model4

    # Load model checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    # Initialize metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # Image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Lists to store metrics
    psnr_values = []
    ssim_values = []
    lpips_values = []
    rci_values = []

    # Prepare directories
    os.makedirs(args.output_dir, exist_ok=True)
    components_dir = os.path.join(args.output_dir, 'components')
    os.makedirs(components_dir, exist_ok=True)

    # Process images
    input_images = sorted(os.listdir(args.test_input_dir))
    gt_images = sorted(os.listdir(args.test_gt_dir))

    # Ensure matching filenames between input and gt
    input_filenames = set(input_images)
    gt_filenames = set(gt_images)
    common_filenames = sorted(input_filenames.intersection(gt_filenames))

    for filename in tqdm(common_filenames, desc='Processing Images'):
        input_path = os.path.join(args.test_input_dir, filename)
        gt_path = os.path.join(args.test_gt_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        # Load images
        input_image = Image.open(input_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')

        # Apply transformations
        input_tensor = transform(input_image).unsqueeze(0).to(device)  # [1, C, H, W]
        gt_tensor = transform(gt_image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Forward pass
            hat_Q1_low, hat_Q2_low = model(input_tensor)

            # Reconstruction using Hamilton product
            recon_quat = hamilton_product(hat_Q1_low, hat_Q2_low)
            recon_images = recon_quat[:, 1:, :, :]  # Imaginary parts correspond to RGB channels

            # Clamp output to [0,1]
            recon_images = torch.clamp(recon_images, 0.0, 1.0)
            gt_images = torch.clamp(gt_tensor, 0.0, 1.0)

        # Save reconstructed image
        output_image = transforms.ToPILImage()(recon_images.squeeze(0).cpu())
        output_image.save(output_path)

        # Compute metrics
        psnr = psnr_metric(recon_images, input_tensor)
        ssim = ssim_metric(recon_images, input_tensor)

        # Normalize tensors to [-1,1] for LPIPS
        def normalize_for_lpips(tensor):
            return tensor * 2 - 1

        output_tensor_lpips = normalize_for_lpips(recon_images)
        gt_tensor_lpips = normalize_for_lpips(input_tensor)
        lpips_value = loss_fn_alex(output_tensor_lpips, gt_tensor_lpips)

        psnr_values.append(psnr.item())
        ssim_values.append(ssim.item())
        lpips_values.append(lpips_value.item())

        # Save components Q1 and Q2
        hat_Q1 = hat_Q1_low.squeeze(0).cpu().numpy()  # [4, H, W]
        hat_Q2 = hat_Q2_low.squeeze(0).cpu().numpy()  # [4, H, W]

        components = ['r', 'i', 'j', 'k']

        # Save Q1 components
        for idx, comp in enumerate(components):
            component = hat_Q1[idx, :, :]
            plt.figure()
            plt.imshow(component, cmap='gray')
            plt.colorbar()
            plt.savefig(os.path.join(components_dir, f'{filename}_Q1_{comp}.png'))
            plt.close()

        # Save Q2 components
        for idx, comp in enumerate(components):
            component = hat_Q2[idx, :, :]
            plt.figure()
            plt.imshow(component, cmap='gray')
            plt.colorbar()
            plt.savefig(os.path.join(components_dir, f'{filename}_Q2_{comp}.png'))
            plt.close()

        # Compute RCI metric if requested
        if args.compute_rci:
            rci = compute_rci(model, input_image, gt_image, device, quaternion=True, save_variance_map=False)
            rci_values.append(rci)

    # Compute average metrics
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_lpips = sum(lpips_values) / len(lpips_values)

    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Average PSNR: {avg_psnr:.4f} dB\n')
        f.write(f'Average SSIM: {avg_ssim:.4f}\n')
        f.write(f'Average LPIPS: {avg_lpips:.4f}\n')
        if args.compute_rci and len(rci_values) > 0:
            avg_rci = sum(rci_values) / len(rci_values)
            f.write(f'Average RCI: {avg_rci:.4f}\n')

    print('Processing completed.')
    print(f'Average PSNR: {avg_psnr:.4f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Average LPIPS: {avg_lpips:.4f}')
    if args.compute_rci and len(rci_values) > 0:
        print(f'Average RCI: {avg_rci:.4f}')

if __name__ == '__main__':
    main()
