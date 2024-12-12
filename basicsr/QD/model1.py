import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

### Wavelet forward and inverse transforms ###
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel = in_batch, in_channel // (r**2)
    out_height, out_width = r * in_height, r * in_width

    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width], device=x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
    def forward(self, x):
        return iwt_init(x)

### New Cross-Attention Module ###
class SymmetricCrossAttention(nn.Module):
    def __init__(self, dim, heads=1):
        super(SymmetricCrossAttention, self).__init__()
        # Keep it simple and lightweight:
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.q1_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k2_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v2_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.q2_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k1_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v1_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.out1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.out2 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, q1_feat, q2_feat):
        # q1_feat, q2_feat: B, C, H, W
        B, C, H, W = q1_feat.shape

        # Compute Q1 from q1_feat, K2 and V2 from q2_feat
        q1 = self.q1_proj(q1_feat)
        k2 = self.k2_proj(q2_feat)
        v2 = self.v2_proj(q2_feat)

        # Compute Q2 from q2_feat, K1 and V1 from q1_feat
        q2 = self.q2_proj(q2_feat)
        k1 = self.k1_proj(q1_feat)
        v1 = self.v1_proj(q1_feat)

        # Reshape for attention: B, heads, C/head, H*W
        def reshape_for_heads(x):
            return rearrange(x, 'b (head c) h w -> b head c (h w)', head=self.heads)

        q1, k2, v2 = map(reshape_for_heads, [q1, k2, v2])
        q2, k1, v1 = map(reshape_for_heads, [q2, k1, v1])

        # Scale queries
        scale = self.head_dim ** -0.5
        q1 = q1 * scale
        q2 = q2 * scale

        # Q1-K2 attention
        attn_1 = torch.softmax(q1 @ k2.transpose(-1, -2), dim=-1) # B, head, C/head, C/head
        cross1 = attn_1 @ v2  # B, head, C/head, H*W

        # Q2-K1 attention
        attn_2 = torch.softmax(q2 @ k1.transpose(-1, -2), dim=-1)
        cross2 = attn_2 @ v1

        # Reshape back
        def reshape_back(x):
            return rearrange(x, 'b head c (h w) -> b (head c) h w', h=H, w=W)

        cross1 = reshape_back(cross1)
        cross2 = reshape_back(cross2)

        # Output projections
        refined_q1 = self.out1(cross1)
        refined_q2 = self.out2(cross2)

        # Combine refined features with original (residual)
        refined_q1 = refined_q1 + q1_feat
        refined_q2 = refined_q2 + q2_feat

        return refined_q1, refined_q2

### Simple Decomposition Network with Sharpening and Cross-Attention ###
class Decomp(nn.Module):
    def __init__(self, 
                 inp_channels=8, # Q1r,Q2r,Q1i,Q2i,Q1j,Q2j,Q1k,Q2k
                 out_channels=8,
                 use_wavelets=True,
                 num_filters=32):
        super(Decomp, self).__init__()
        self.use_wavelets = use_wavelets
        if self.use_wavelets:
            self.dwt = DWT()
            self.iwt = IWT()
            inp_channels *= 4
            out_channels *= 4

        # A simple shallow feature extractor
        self.conv_in = nn.Conv2d(inp_channels, num_filters, kernel_size=3, padding=1)
        
        # Separate branches for Q1 and Q2 refinement
        self.branch_q1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1)
        )

        self.branch_q2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1)
        )

        # Introduce symmetrical cross-attention between Q1 and Q2
        self.cross_attn = SymmetricCrossAttention(num_filters)

        # Fuse Q1 and Q2
        self.fuse = nn.Conv2d(num_filters * 2, num_filters, kernel_size=1)
        self.conv_out = nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1)
        
        # Sharpening block
        self.sharpening = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.constant_(self.sharpening.bias, 0)
        with torch.no_grad():
            laplacian = torch.tensor([[[0, -1, 0],
                                       [-1, 4, -1],
                                       [0, -1, 0]]], dtype=torch.float)
            laplacian = laplacian.expand(out_channels, out_channels, 3, 3) / out_channels
            self.sharpening.weight.copy_(laplacian)

    def forward(self, inp_img):
        eps = 1e-7
        # Convert to quaternion form Q1 and Q2:
        R, G, B = torch.split(inp_img, 1, dim=1)
        rgbMax = torch.max(inp_img, dim=1, keepdim=True)[0]
        q1_r = torch.zeros_like(rgbMax)
        q1_i = R / (rgbMax + eps)
        q1_j = G / (rgbMax + eps)
        q1_k = B / (rgbMax + eps)

        q2_r = torch.zeros_like(rgbMax)
        q2_i = R
        q2_j = G
        q2_k = B

        input_tensor = torch.cat([q1_r, q2_r, 
                                  q1_i, q2_i, 
                                  q1_j, q2_j, 
                                  q1_k, q2_k], dim=1)

        if self.use_wavelets:
            input_tensor = self.dwt(input_tensor)

        feat = self.conv_in(input_tensor)

        # Q1 and Q2 branches
        q1_feat = self.branch_q1(feat) + feat
        q2_feat = self.branch_q2(feat) + feat

        # Cross-attention refinement between Q1 and Q2
        q1_feat, q2_feat = self.cross_attn(q1_feat, q2_feat)

        fused_feat = torch.cat([q1_feat, q2_feat], dim=1)
        fused_feat = self.fuse(fused_feat)
        out = self.conv_out(fused_feat)
        
        # Apply sharpening
        sharp_map = self.sharpening(out)
        out = out + sharp_map  # Add high-frequency details

        if self.use_wavelets:
            out = self.iwt(out)

        # Split back into q1 and q2
        q1 = out[:, [0, 2, 4, 6], :, :]
        q2 = out[:, [1, 3, 5, 7], :, :]

        return q1, q2