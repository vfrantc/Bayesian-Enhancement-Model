import torch
import torch.nn as nn
import torch.nn.functional as F
from quaternion import hamilton_product

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

        # According to RetinexNet decomposition loss:
        # recon_loss_low  = L1(recon_low, input_low)
        # recon_loss_high = L1(recon_high, input_high)
        # recon_loss_mutal_low  = L1(recon_mutal_low, input_low)
        # recon_loss_mutal_high = L1(recon_mutal_high, input_high)
        # equal_R_loss = L1(R_low, R_high.detach())
        # Ismooth_loss_low  = smooth(I_low,  R_low)
        # Ismooth_loss_high = smooth(I_high, R_high)

        recon_loss_low = F.l1_loss(recon_low, input_low)
        recon_loss_high = F.l1_loss(recon_high, input_high)
        recon_loss_mutal_low = F.l1_loss(recon_mutal_low, input_low)
        recon_loss_mutal_high = F.l1_loss(recon_mutal_high, input_high)
        equal_R_loss = F.l1_loss(R_low, R_high.detach())

        Ismooth_loss_low = self.smooth(I_low, R_low)
        Ismooth_loss_high = self.smooth(I_high, R_high)

        # Final decomposition loss:
        loss_Decom = recon_loss_low + \
                     recon_loss_high + \
                     0.001 * recon_loss_mutal_low + \
                     0.001 * recon_loss_mutal_high + \
                     0.1 * Ismooth_loss_low + \
                     0.1 * Ismooth_loss_high + \
                     0.01 * equal_R_loss

        return {
            'loss_Decom': loss_Decom,
            'recon_loss_low': recon_loss_low,
            'recon_loss_high': recon_loss_high,
            'recon_loss_mutal_low': recon_loss_mutal_low,
            'recon_loss_mutal_high': recon_loss_mutal_high,
            'equal_R_loss': equal_R_loss,
            'Ismooth_loss_low': Ismooth_loss_low,
            'Ismooth_loss_high': Ismooth_loss_high
        }
