import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataset import PairDataset
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from quaternion import hamilton_product  # Ensure quaternion.py is in the same directory or properly installed

import torch
import torch.nn as nn
import torch.nn.functional as F
from quaternion import hamilton_product

def frequency_regularization(img, weight=0.01):
    # Compute the Fourier Transform
    fft = torch.fft.fft2(img, norm="ortho")
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    fft_abs = torch.abs(fft_shift)
    
    # Penalize high-frequency components (assuming high frequencies are at the corners after fftshift)
    # Modify the range [-10:, -10:] based on your specific requirements
    freq_loss = torch.mean(fft_abs[:, :, -10:, -10:])
    return weight * freq_loss

def total_variation_loss(img, weight=0.1):
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return weight * (tv_h + tv_w)

class RetinexLoss(nn.Module):
    def __init__(self):
        super(RetinexLoss, self).__init__()
        # Predefine kernels for smoothness
        self.register_buffer('smooth_kernel_x', torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)))
        self.register_buffer('smooth_kernel_y', torch.FloatTensor([[0, -1], [0, 1]]).view((1, 1, 2, 2)))

    def gradient(self, input_tensor, direction):
        # direction: "x" or "y"
        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        # input_I and input_R are [B,3,H,W] images (imaginary parts of quaternion)
        # Convert R to grayscale
        R_gray = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        R_gray = R_gray.unsqueeze(1)  # [B,1,H,W]

        # Convert I to grayscale
        I_gray = 0.299*input_I[:, 0, :, :] + 0.587*input_I[:, 1, :, :] + 0.114*input_I[:, 2, :, :]
        I_gray = I_gray.unsqueeze(1)  # [B,1,H,W]

        # Apply gradients on single-channel tensors
        return torch.mean(
            self.gradient(I_gray, "x") * torch.exp(-10 * self.ave_gradient(R_gray, "x")) +
            self.gradient(I_gray, "y") * torch.exp(-10 * self.ave_gradient(R_gray, "y"))
        )

    def forward(self, Q1_low, Q2_low, input_low, Q1_high, Q2_high, input_high):
        # Q1: reflectance quaternion [B,4,H,W]
        # Q2: illumination quaternion [B,4,H,W]
        
        # Extract imaginary parts (RGB channels) of Q1 and Q2
        R_low = Q1_low[:, 1:, :, :]  # [B,3,H,W]
        I_low = Q2_low[:, 1:, :, :]  # [B,3,H,W]
        R_high = Q1_high[:, 1:, :, :]
        I_high = Q2_high[:, 1:, :, :]

        # Compute reconstructed images using Hamilton product (Q1 * Q2), then take imaginary part
        recon_low = hamilton_product(Q1_low, Q2_low)[:, 1:, :, :]    # [B,3,H,W]
        recon_high = hamilton_product(Q1_high, Q2_high)[:, 1:, :, :] # [B,3,H,W]

        # Mutual reconstructions
        recon_mutal_low = hamilton_product(Q1_high, Q2_low)[:, 1:, :, :]   # [B,3,H,W]
        recon_mutal_high = hamilton_product(Q1_low, Q2_high)[:, 1:, :, :]  # [B,3,H,W]

        # Reconstruction Losses
        recon_loss_low = F.l1_loss(recon_low, input_low)
        recon_loss_high = F.l1_loss(recon_high, input_high)
        recon_loss_mutal_low = F.l1_loss(recon_mutal_low, input_low)
        recon_loss_mutal_high = F.l1_loss(recon_mutal_high, input_high)
        equal_R_loss = F.l1_loss(R_low, R_high.detach())

        # Smoothness Losses
        Ismooth_loss_low = self.smooth(I_low, R_low) + total_variation_loss(R_low)
        Ismooth_loss_high = self.smooth(I_high, R_high) + total_variation_loss(R_high)

        # Frequency Regularization Loss
        freq_loss = frequency_regularization(recon_low) + frequency_regularization(recon_high)

        # Total Decomposition Loss with Balanced Weights
        loss_Decom = recon_loss_low + \
                     recon_loss_high + \
                     0.01 * recon_loss_mutal_low + \
                     0.01 * recon_loss_mutal_high + \
                     0.05 * Ismooth_loss_low + \
                     0.05 * Ismooth_loss_high + \
                     0.01 * equal_R_loss + \
                     freq_loss  # You can adjust the weight of freq_loss if needed

        # Optionally, include TV loss separately if needed
        # tv_loss = total_variation_loss(R_low) + total_variation_loss(R_high)
        # loss_Decom += tv_loss

        return {
            'loss_Decom': loss_Decom,
            'recon_loss_low': recon_loss_low,
            'recon_loss_high': recon_loss_high,
            'recon_loss_mutal_low': recon_loss_mutal_low,
            'recon_loss_mutal_high': recon_loss_mutal_high,
            'equal_R_loss': equal_R_loss,
            'Ismooth_loss_low': Ismooth_loss_low,
            'Ismooth_loss_high': Ismooth_loss_high,
            'freq_loss': freq_loss,
            # 'tv_loss': tv_loss  # Uncomment if TV loss is separately logged
        }


def main():
    parser = argparse.ArgumentParser(description='Training script for low-light image enhancement with Retinex-based quaternion decomposition.')
    parser.add_argument('--model_type', type=str, choices=['model1', 'model2', 'model3', 'model4'], default='model1')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr_min', type=float, default=1e-7)
    parser.add_argument('--lr_max', type=float, default=3e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10)  # Increased from 5 to 10
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--train_input_dir', type=str, default='../../data/train/LOL/input')
    parser.add_argument('--train_gt_dir', type=str, default='../../data/train/LOL/gt')
    parser.add_argument('--test_input_dir', type=str, default='../../data/test/LOL/input')
    parser.add_argument('--test_gt_dir', type=str, default='../../data/test/LOL/gt')
    parser.add_argument('--log_dir', type=str, default='./runs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--eval_dir', type=str, default='./evaluation_results')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--evaluate_script', type=str, default='evaluate.py')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use for training.')

    args = parser.parse_args()

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Create model-specific directories
    model_log_dir = os.path.join(args.log_dir, args.model_type)
    model_checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_type)
    model_eval_dir = os.path.join(args.eval_dir, args.model_type)

    os.makedirs(model_checkpoint_dir, exist_ok=True)
    os.makedirs(model_log_dir, exist_ok=True)
    os.makedirs(model_eval_dir, exist_ok=True)

    writer = SummaryWriter(logdir=model_log_dir)

    patch_size = args.patch_size
    num_epochs = args.num_epochs
    lr_min = args.lr_min
    lr_max = args.lr_max
    warmup_epochs = args.warmup_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers

    train_dataset = PairDataset(args.train_input_dir, args.train_gt_dir, patch_size=patch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = PairDataset(args.test_input_dir, args.test_gt_dir, patch_size=patch_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Select model
    if args.model_type == 'model1':
        from model1 import Decomp
        model = Decomp().to(device)
    elif args.model_type == 'model2':
        from model2 import Decomp as Decomp2
        model = Decomp2().to(device)
    elif args.model_type == 'model3':
        from model3 import Decomp as Decomp3
        model = Decomp3().to(device)
    elif args.model_type == 'model4':
        from model4 import Decomp as Decomp4
        model = Decomp4().to(device)

    print(f'Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.999))

    # Updated Learning Rate Scheduler: Cyclical Learning Rate
    from torch.optim.lr_scheduler import CyclicLR, SequentialLR, CosineAnnealingLR, LinearLR
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=lr_min)
    scheduler_linear = LinearLR(optimizer, start_factor=1.0, total_iters=warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_linear, scheduler_cosine], milestones=[warmup_epochs])
    # Alternatively, if you prefer CyclicLR over SequentialLR, uncomment the following lines:
    # scheduler = CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max, step_size_up=200, mode='triangular')

    scheduler.step()

    retinex_loss_fn = RetinexLoss().to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    start_epoch = 0
    if args.resume:
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from checkpoint: {args.checkpoint_path}")
        else:
            print("No checkpoint path specified for resuming.")
            exit(1)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss_total = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch [{epoch + 1}/{num_epochs}], Current Learning Rate: {current_lr:.6f}')

        # Adjust patch size progressively
        if epoch < 100:
            current_patch_size = 64
        elif epoch < 500:
            current_patch_size = 128
        else:
            current_patch_size = 256
        # Optionally, you can adjust the dataset patch size here if your dataset class supports dynamic patch sizes

        for i, (input_images, gt_images) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
            input_images, gt_images = input_images.to(device), gt_images.to(device)

            # Forward pass
            hat_Q1_low, hat_Q2_low = model(input_images)
            hat_Q1_high, hat_Q2_high = model(gt_images)

            # Compute Retinex decomposition losses
            loss_dict = retinex_loss_fn(hat_Q1_low, hat_Q2_low, input_images,
                                        hat_Q1_high, hat_Q2_high, gt_images)

            # Apply different loss weighting during warmup
            if epoch < 50:
                loss_total = loss_dict['recon_loss_low'] + loss_dict['recon_loss_high']
            else:
                loss_total = loss_dict['loss_Decom']

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            running_loss_total += loss_total.item()
            global_step = epoch * len(train_loader) + i

            # Log losses
            writer.add_scalar('Loss/Total', loss_total.item(), global_step)
            writer.add_scalar('Loss/Recon_Low', loss_dict.get('recon_loss_low', 0).item(), global_step)
            writer.add_scalar('Loss/Recon_High', loss_dict.get('recon_loss_high', 0).item(), global_step)
            writer.add_scalar('Loss/Recon_Mutual_Low', loss_dict.get('recon_loss_mutal_low', 0).item(), global_step)
            writer.add_scalar('Loss/Recon_Mutual_High', loss_dict.get('recon_loss_mutal_high', 0).item(), global_step)
            writer.add_scalar('Loss/Smoothness_Low', loss_dict.get('Ismooth_loss_low', 0).item(), global_step)
            writer.add_scalar('Loss/Smoothness_High', loss_dict.get('Ismooth_loss_high', 0).item(), global_step)
            writer.add_scalar('Loss/Equal_R', loss_dict.get('equal_R_loss', 0).item(), global_step)
            writer.add_scalar('Loss/Frequency', loss_dict.get('freq_loss', 0).item(), global_step)
            writer.add_scalar('Loss/TV', loss_dict.get('tv_loss', 0), global_step)  # If TV loss is added

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Total Loss: {loss_total.item():.4f}')

        avg_loss_total = running_loss_total / len(train_loader)
        writer.add_scalar('Epoch_Loss/Train', avg_loss_total, epoch)
        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Total Loss: {avg_loss_total:.4f}')

        # Validation
        model.eval()
        validation_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0

        with torch.no_grad():
            for input_images, gt_images in tqdm(test_loader, desc='Validation'):
                input_images, gt_images = input_images.to(device), gt_images.to(device)
                hat_Q1_low, hat_Q2_low = model(input_images)
                # Reconstruct using Hamilton product
                recon_quat = hamilton_product(hat_Q1_low, hat_Q2_low)
                recon_images = recon_quat[:, 1:, :, :]  # Imag parts as RGB
                recon_images = torch.clamp(recon_images, 0.0, 1.0)
                input_images = torch.clamp(input_images, 0.0, 1.0)

                loss = torch.nn.functional.l1_loss(recon_images, input_images)
                validation_loss += loss.item()

                psnr = psnr_metric(recon_images, input_images)
                ssim = ssim_metric(recon_images, input_images)

                total_psnr += psnr.item()
                total_ssim += ssim.item()

        avg_validation_loss = validation_loss / len(test_loader)
        avg_psnr = total_psnr / len(test_loader)
        avg_ssim = total_ssim / len(test_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_validation_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}')
        writer.add_scalar('Epoch_Loss/Validation', avg_validation_loss, epoch)
        writer.add_scalar('Metrics/PSNR', avg_psnr, epoch)
        writer.add_scalar('Metrics/SSIM', avg_ssim, epoch)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, f'{model_checkpoint_dir}/model_epoch_{epoch + 1}.pth')

    writer.close()

    # Run evaluation
    final_checkpoint_path = f'{model_checkpoint_dir}/model_epoch_{num_epochs}.pth'
    eval_command = (
        f'python {args.evaluate_script} '
        f'--model_checkpoint {final_checkpoint_path} '
        f'--test_input_dir {args.test_input_dir} '
        f'--test_gt_dir {args.test_gt_dir} '
        f'--output_dir {model_eval_dir} '
        f'--device {device.type if device.type=="cpu" else "cuda:0"} '
        f'--model_type {args.model_type} '
    )
    print("Running evaluation...")
    os.system(eval_command)

if __name__ == '__main__':
    main()
