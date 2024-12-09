import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import math
import warnings
from basicsr.archs.arch_util import SAM
from basicsr.utils.registry import ARCH_REGISTRY
try:
    from ..vmamba.models.vmamba import VSSBlock, LayerNorm2d
except:
    from vmamba.models.vmamba import VSSBlock, LayerNorm2d
from functools import partial


def _no_grad_trunc_normal_(
    tensor, mean, std, a, b
):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(
    tensor, mean=0.0, std=1.0, a=-2.0, b=2.0
):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def conv_down(in_channels):
    return nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)

def deconv_up(in_channels):
    return nn.ConvTranspose2d(
        in_channels,
        in_channels // 2,
        stride=2,
        kernel_size=2,
        padding=0,
        output_padding=0,
    )


class WindowedCrossAttention(nn.Module):
    """
    A memory-optimized cross-attention module that operates on local windows.
    This helps reduce the OOM issue encountered with large feature maps.
    """
    def __init__(self, embed_dim, num_heads=2, window_size=16, downsample_factor=1):
        super(WindowedCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (embed_dim % num_heads) == 0
        self.scale = self.head_dim ** -0.5

        self.query = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.key = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.value = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.window_size = window_size
        self.downsample_factor = downsample_factor
        if downsample_factor > 1:
            self.downsample = nn.AvgPool2d(downsample_factor, downsample_factor)
            self.upsample = nn.Upsample(scale_factor=downsample_factor, mode='nearest')
        else:
            self.downsample = None
            self.upsample = None

    def forward(self, x_q, x_kv):
        B, C, H, W = x_q.shape
        # Optionally downsample to reduce memory
        if self.downsample is not None:
            x_q = self.downsample(x_q)
            x_kv = self.downsample(x_kv)

        B, C, H_ds, W_ds = x_q.shape

        q = self.query(x_q)  # B,C,H_ds,W_ds
        k = self.key(x_kv)
        v = self.value(x_kv)

        # Break feature maps into windows
        # Assume H_ds,W_ds are multiples of window_size for simplicity.
        win = self.window_size
        # Reshape into windows: (B, C, num_winH, win, num_winW, win)
        q = q.reshape(B, C, H_ds//win, win, W_ds//win, win).permute(0,2,4,1,3,5) # B, nH, nW, C, win, win
        k = k.reshape(B, C, H_ds//win, win, W_ds//win, win).permute(0,2,4,1,3,5)
        v = v.reshape(B, C, H_ds//win, win, W_ds//win, win).permute(0,2,4,1,3,5)

        # Now we have windows as (B, nH, nW, C, win, win)
        # Reshape per window:
        # Flatten spatial for q,k,v inside each window
        nH = H_ds//win
        nW = W_ds//win

        q = q.reshape(B*nH*nW, self.num_heads, self.head_dim, win*win).permute(0,1,3,2) # B*nH*nW, heads, win^2, head_dim
        k = k.reshape(B*nH*nW, self.num_heads, self.head_dim, win*win)
        v = v.reshape(B*nH*nW, self.num_heads, self.head_dim, win*win).permute(0,1,3,2)

        # q: B*nH*nW, heads, win^2, head_dim
        # k: B*nH*nW, heads, head_dim, win^2
        # v: B*nH*nW, heads, win^2, head_dim
        attn = (q @ k) * self.scale  # (B*nH*nW, heads, win^2, win^2)
        attn = attn.softmax(dim=-1)

        out = attn @ v # (B*nH*nW, heads, win^2, head_dim)
        out = out.permute(0,1,3,2).reshape(B*nH*nW, C, win, win)

        # Merge windows back
        out = out.reshape(B, nH, nW, C, win, win).permute(0,3,1,4,2,5).reshape(B,C,H_ds,W_ds)

        out = self.out(out)

        # Upsample if downsampled
        if self.upsample is not None:
            out = self.upsample(out)

        return out


@ARCH_REGISTRY.register()
class TwoBranchVMUNet(nn.Module):
    """
    A two-branch U-Net style architecture with a cross-attention layer optimized for memory:
    - Uses window-based cross-attention to reduce OOM issues.
    - Incorporates a two-branch decomposition and recomposition approach.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=40,
        stage=1,
        num_blocks=[2, 2, 2],
        d_state=1,
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        use_pixelshuffle=False,
        last_act=None,
        drop_path=0,
        sam=0,
        branch1_in_channels=2,
        branch2_in_channels=1,
        cross_attention_dim=40,
        # Additional config for memory optimization
        cross_attention_heads=2,
        cross_window_size=16,
        cross_downsample=1
    ):
        super(TwoBranchVMUNet, self).__init__()
        self.stage = stage
        self.num_levels = len(num_blocks)
        if isinstance(d_state, int):
            d_state = [d_state]*self.num_levels

        self.out_channels = out_channels
        self.last_act_type = last_act
        if last_act is None:
            self.last_act = nn.Identity()
        elif last_act == "relu":
            self.last_act = nn.ReLU()
        elif last_act == "softmax":
            self.last_act = nn.Softmax(dim=1)
        else:
            raise NotImplementedError

        if use_pixelshuffle:
            down_layer = conv_down
            up_layer = deconv_up
        else:
            down_layer = conv_down
            up_layer = deconv_up

        # ========== Branch 1 ==========
        self.branch1_first_conv = nn.Conv2d(branch1_in_channels, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(self.branch1_first_conv.weight, mode="fan_out", nonlinearity="linear")
        if self.branch1_first_conv.bias is not None:
            nn.init.zeros_(self.branch1_first_conv.bias)

        self.branch1_encoders = nn.ModuleList()
        curr_dim = n_feat
        for i in range(self.num_levels - 1):
            self.branch1_encoders.append(self._make_level(curr_dim, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type))
            curr_dim *= 2
        self.branch1_bottleneck = self._make_level(curr_dim, num_blocks[-1], d_state[-1], ssm_ratio, mlp_ratio, mlp_type)
        self.branch1_decoders = nn.ModuleList()
        for i in range(self.num_levels - 2, -1, -1):
            self.branch1_decoders.append(nn.ModuleDict({
                'up': up_layer(curr_dim),
                'fuse': nn.Conv2d(curr_dim, curr_dim // 2, 1, 1, bias=False),
                'block': self._make_level(curr_dim // 2, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type)
            }))
            curr_dim //= 2

        self.branch1_proj = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)
        nn.init.zeros_(self.branch1_proj.bias)
        self.branch1_down_layers = nn.ModuleList([down_layer(n_feat * (2 ** i)) for i in range(self.num_levels - 1)])

        # ========== Branch 2 ==========
        self.branch2_first_conv = nn.Conv2d(branch2_in_channels, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(self.branch2_first_conv.weight, mode="fan_out", nonlinearity="linear")
        if self.branch2_first_conv.bias is not None:
            nn.init.zeros_(self.branch2_first_conv.bias)

        self.branch2_encoders = nn.ModuleList()
        curr_dim = n_feat
        for i in range(self.num_levels - 1):
            self.branch2_encoders.append(self._make_level(curr_dim, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type))
            curr_dim *= 2
        self.branch2_bottleneck = self._make_level(curr_dim, num_blocks[-1], d_state[-1], ssm_ratio, mlp_ratio, mlp_type)
        self.branch2_decoders = nn.ModuleList()
        for i in range(self.num_levels - 2, -1, -1):
            self.branch2_decoders.append(nn.ModuleDict({
                'up': up_layer(curr_dim),
                'fuse': nn.Conv2d(curr_dim, curr_dim // 2, 1, 1, bias=False),
                'block': self._make_level(curr_dim // 2, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type)
            }))
            curr_dim //= 2

        self.branch2_proj = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)
        nn.init.zeros_(self.branch2_proj.bias)
        self.branch2_down_layers = nn.ModuleList([down_layer(n_feat * (2 ** i)) for i in range(self.num_levels - 1)])

        # Cross-attention (windowed)
        self.cross_attention = WindowedCrossAttention(
            cross_attention_dim, 
            num_heads=cross_attention_heads, 
            window_size=cross_window_size,
            downsample_factor=cross_downsample
        )

        # Refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(n_feat*2, n_feat, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, out_channels, 3, 1, 1, bias=True)
        )
        if self.refinement[-1].bias is not None:
            nn.init.zeros_(self.refinement[-1].bias)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _make_level(self, dim, num_block, d_state, ssm_ratio, mlp_ratio, mlp_type):
        layers = []
        for _ in range(num_block):
            layers.append(
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=0,
                    norm_layer=LayerNorm2d,
                    channel_first=True,
                    ssm_d_state=d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank="auto",
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=3,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0,
                    ssm_init="v0",
                    forward_type="v05_noz",
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    mlp_type=mlp_type,
                    use_checkpoint=False,
                    post_norm=False,
                )
            )
        return nn.Sequential(*layers)

    def decompose_fn(self, x):
        # Dummy decomposition into H, V, I
        r = x[:,0:1]
        g = x[:,1:2]
        b = x[:,2:3]

        H = 0.299*r + 0.587*g + 0.114*b
        V = r - b
        I = g

        branch1_in = torch.cat([H, V], dim=1)
        branch2_in = I
        return branch1_in, branch2_in, (H, V, I)

    def recompose_fn(self, H, V, I):
        # Placeholder recompose (not exact inverse)
        R = I + V
        G = I
        B = I
        return torch.cat([R, G, B], dim=1)

    def forward_single_branch(self, x, encoders, bottleneck, decoders, down_layers):
        fea = x
        skip_connections = []
        # Encoding
        for i, enc in enumerate(encoders):
            fea = enc(fea)
            skip_connections.append(fea)
            fea = down_layers[i](fea)
        # Bottleneck
        fea = bottleneck(fea)

        # Decoding
        for i, dec in enumerate(decoders):
            skip = skip_connections[len(decoders)-1-i]
            fea = dec['up'](fea)
            fea = dec['fuse'](torch.cat([fea, skip], dim=1))
            fea = dec['block'](fea)
        return fea

    def forward(self, x, mask=None):
        # Decompose input into H, V, I and feed into two branches
        branch1_in, branch2_in, (H, V, I) = self.decompose_fn(x)

        # Forward branch 1
        branch1_fea = self.branch1_first_conv(branch1_in)
        branch1_out = self.forward_single_branch(
            branch1_fea, 
            self.branch1_encoders, 
            self.branch1_bottleneck, 
            self.branch1_decoders, 
            self.branch1_down_layers
        )
        branch1_out = self.branch1_proj(branch1_out)

        # Forward branch 2
        branch2_fea = self.branch2_first_conv(branch2_in)
        branch2_out = self.forward_single_branch(
            branch2_fea, 
            self.branch2_encoders, 
            self.branch2_bottleneck, 
            self.branch2_decoders, 
            self.branch2_down_layers
        )
        branch2_out = self.branch2_proj(branch2_out)

        # Cross-attention between the two branches
        combined1 = self.cross_attention(branch1_out, branch2_out)
        combined2 = self.cross_attention(branch2_out, branch1_out)

        # Fuse the two combined features
        fused = torch.cat([combined1, combined2], dim=1)

        # Refinement - now let's assume the refinement output is in H, V, I space
        out = self.refinement(fused)  # out: B x 3 x H x W
        # Assume out[:,0:1] = H_pred, out[:,1:2] = V_pred, out[:,2:3] = I_pred
        H_pred = out[:, 0:1]
        V_pred = out[:, 1:2]
        I_pred = out[:, 2:3]

        # Recompose to get final RGB output
        final_rgb = self.recompose_fn(H_pred, V_pred, I_pred)
        
        # Apply last activation if any
        final_rgb = self.last_act(final_rgb)

        return [x, final_rgb]
