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

### Utility blocks ###
class ReflectionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super(ReflectionConvBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, groups=groups)
        self.norm = nn.GroupNorm(1, out_channels)  # GroupNorm(1, x) ~ LayerNorm in some sense, keeping it here
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ReflectionConvNoAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super(ReflectionConvNoAct, self).__init__()
        self.pad = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, groups=groups)
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        return x

### Symmetric Cross-Attention Module with an option for normalization ###
class SymmetricCrossAttention(nn.Module):
    def __init__(self, dim, heads=1, dropout=0.1):
        super(SymmetricCrossAttention, self).__init__()
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

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q1_feat, q2_feat):
        B, C, H, W = q1_feat.shape

        # Compute Q1 from q1_feat, K2 and V2 from q2_feat
        q1 = self.q1_proj(q1_feat)
        k2 = self.k2_proj(q2_feat)
        v2 = self.v2_proj(q2_feat)

        # Compute Q2 from q2_feat, K1 and V1 from q1_feat
        q2 = self.q2_proj(q2_feat)
        k1 = self.k1_proj(q1_feat)
        v1 = self.v1_proj(q1_feat)

        def reshape_for_heads(x):
            return rearrange(x, 'b (head c) h w -> b head c (h w)', head=self.heads)

        q1, k2, v2 = map(reshape_for_heads, [q1, k2, v2])
        q2, k1, v1 = map(reshape_for_heads, [q2, k1, v1])

        scale = self.head_dim ** -0.5
        q1 = q1 * scale
        q2 = q2 * scale

        # Q1-K2 attention
        attn_1 = torch.softmax(q1 @ k2.transpose(-1, -2), dim=-1)
        attn_1 = self.attn_drop(attn_1)
        cross1 = attn_1 @ v2

        # Q2-K1 attention
        attn_2 = torch.softmax(q2 @ k1.transpose(-1, -2), dim=-1)
        attn_2 = self.attn_drop(attn_2)
        cross2 = attn_2 @ v1

        def reshape_back(x):
            return rearrange(x, 'b head c (h w) -> b (head c) h w', h=H, w=W)

        cross1 = reshape_back(cross1)
        cross2 = reshape_back(cross2)

        refined_q1 = self.out1(cross1) + q1_feat
        refined_q2 = self.out2(cross2) + q2_feat

        return refined_q1, refined_q2

### Simple Feed-Forward Network (Position-Independent) ###
class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super(FeedForward, self).__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

### A small smoothing block for Q2 to reduce grid effects ###
class SmoothBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(SmoothBlock, self).__init__()
        self.smooth = nn.Sequential(
            ReflectionConvNoAct(channels, channels, kernel_size, groups=channels),
            nn.ReLU(inplace=True),
            ReflectionConvNoAct(channels, channels, kernel_size, groups=channels)
        )

    def forward(self, x):
        return x + self.smooth(x)

class Decomp(nn.Module):
    def __init__(self,
                 inp_channels=8,
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

        self.conv_in = nn.Conv2d(inp_channels, num_filters, kernel_size=3, padding=1)

        # A lightweight down-up path (like a mini U-Net)
        self.down_conv = nn.Conv2d(num_filters, num_filters, 3, padding=1, stride=2)
        self.mid_conv = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.up_conv = nn.ConvTranspose2d(num_filters, num_filters, 2, stride=2)

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

        self.cross_attn = SymmetricCrossAttention(num_filters)

        self.fuse = nn.Conv2d(num_filters * 2, num_filters, kernel_size=1)
        self.conv_out = nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1)

        # Sharpening
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
        
        # Downsample
        down_feat = self.down_conv(feat)
        down_feat = F.relu(down_feat, inplace=True)
        
        mid_feat = self.mid_conv(down_feat)
        mid_feat = F.relu(mid_feat, inplace=True)
        
        # Upsample back
        up_feat = self.up_conv(mid_feat)
        # Skip connection
        feat = feat + up_feat

        q1_feat = self.branch_q1(feat) + feat
        q2_feat = self.branch_q2(feat) + feat

        q1_feat, q2_feat = self.cross_attn(q1_feat, q2_feat)

        fused_feat = torch.cat([q1_feat, q2_feat], dim=1)
        fused_feat = self.fuse(fused_feat)
        out = self.conv_out(fused_feat)

        sharp_map = self.sharpening(out)
        out = out + sharp_map

        if self.use_wavelets:
            out = self.iwt(out)

        q1 = out[:, [0, 2, 4, 6], :, :]
        q2 = out[:, [1, 3, 5, 7], :, :]

        return q1, q2