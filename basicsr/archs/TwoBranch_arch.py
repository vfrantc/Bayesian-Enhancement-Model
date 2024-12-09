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


class CrossAttention(nn.Module):
    """A simple cross-attention layer that takes queries from one feature map 
    and keys/values from another feature map."""
    def __init__(self, embed_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (embed_dim % num_heads) == 0
        self.scale = self.head_dim ** -0.5

        self.query = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.key = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.value = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x_q, x_kv):
        B, C, H, W = x_q.shape
        q = self.query(x_q)  # B,C,H,W
        k = self.key(x_kv)   # B,C,H,W
        v = self.value(x_kv) # B,C,H,W

        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        q = q.permute(0,1,3,2) # B,heads,HW,head_dim
        k = k.permute(0,1,2,3) # B,heads,head_dim,HW

        attn = (q @ k) * self.scale  # B,heads,HW,HW
        attn = attn.softmax(dim=-1)

        v = v.permute(0,1,3,2) # B,heads,HW,head_dim
        out = attn @ v # B,heads,HW,head_dim
        out = out.permute(0,1,3,2).reshape(B, C, H, W)
        out = self.out(out)
        return out


@ARCH_REGISTRY.register()
class TwoBranchVMUNet(nn.Module):
    """
    A two-branch U-Net style architecture that:
    - Decomposes the input image into multiple components (e.g. H, V, I).
    - Processes them with two VMUNet-like branches.
    - Uses a cross-attention layer in the middle to exchange information.
    - Then refines and recomposes the output.

    This is designed so that the decomposition/recomposition steps can be changed easily in the future.
    The main structure is similar to VMUNet but now we have two sets of encoders/decoders and a cross-attention block.
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
        # Additional parameters for two-branch structure
        # Suppose we decompose image into H, V, I (3 channels) and split them into two branches:
        # branch1_in_channels and branch2_in_channels define how we split.
        # For example, if we have 3 channels (H,V,I) and we want two branches:
        #   branch1 gets H,V (2 channels)
        #   branch2 gets I (1 channel)
        branch1_in_channels=2,
        branch2_in_channels=1,
        cross_attention_dim=40,  # dimension at which to apply cross-attention
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

        # Define down/up sampling
        if use_pixelshuffle:
            down_layer = conv_down
            up_layer = deconv_up
        else:
            down_layer = conv_down
            up_layer = deconv_up

        # ========== Branch 1 ==========
        # First conv for branch 1
        self.branch1_first_conv = nn.Conv2d(branch1_in_channels, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(self.branch1_first_conv.weight, mode="fan_out", nonlinearity="linear")
        if self.branch1_first_conv.bias is not None:
            nn.init.zeros_(self.branch1_first_conv.bias)

        # Encoders for branch 1
        self.branch1_encoders = nn.ModuleList()
        curr_dim = n_feat
        for i in range(self.num_levels - 1):
            self.branch1_encoders.append(self._make_level(curr_dim, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type))
            curr_dim *= 2

        # Bottleneck for branch 1
        self.branch1_bottleneck = self._make_level(curr_dim, num_blocks[-1], d_state[-1], ssm_ratio, mlp_ratio, mlp_type)

        # Decoders for branch 1
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

        # Cross-attention after bottleneck
        self.cross_attention = CrossAttention(cross_attention_dim, num_heads=4)

        # Refinement layer after recombining the two branches
        # Let's do a simple conv block as refinement
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
        """
        Decompose the input into two branches.
        For demonstration, we assume we split into H, V, I from an RGB image.
        In practice, you can replace this with any decomposition.
        
        Here, let's say we want:
        - H, V from the first two channels (just a placeholder)
        - I from the last channel

        If the input is RGB: we can convert to some space (like HVI).
        For now, let's simulate HVI as a dummy decomposition:

        H = 0.299*R + 0.587*G + 0.114*B
        V = R - B  (just a placeholder)
        I = G       (just a placeholder)
        """
        r = x[:,0:1]
        g = x[:,1:2]
        b = x[:,2:3]

        # Dummy decomposition:
        H = 0.299*r + 0.587*g + 0.114*b
        V = r - b
        I = g

        # Now assign them to two branches:
        # Branch1 = H,V (2-channels)
        # Branch2 = I   (1-channel)
        branch1_in = torch.cat([H, V], dim=1)
        branch2_in = I
        return branch1_in, branch2_in, (H, V, I)

    def recompose_fn(self, H, V, I):
        """
        Recompose from H, V, I back to RGB or the final desired color space.
        This is a placeholder inverse of the dummy decomposition above.
        
        If:
        H = 0.299R + 0.587G + 0.114B
        V = R - B
        I = G

        Let's solve a simple system (this is just for demonstration):
        I = G
        H = 0.299R + 0.587G + 0.114B
        V = R - B

        From I = G, we get G = I.
        From V = R - B, let R = V + B.
        Substitute R into H:
        H = 0.299(V+B) + 0.587I + 0.114B
          = 0.299V + (0.299+0.114)*B + 0.587I
          = 0.299V + 0.413B + 0.587I

        We have two unknowns V and B in H. This is not a well-defined system.
        For simplicity, let's assume B = I (just a placeholder guess).
        If B = I, then R = V + I.

        Then:
        H ≈ 0.299V + 0.413I + 0.587I
          = 0.299V + 1.0I
        => 0.299V = H - I
        V = (H - I)/0.299

        This gets complicated, but since this is just a placeholder, 
        we'll just do a simple linear combination to get back some RGB-like output:
        Let's just do a trivial recomposition:
        R = I + V
        G = I
        B = I

        This is not a true inverse, but a placeholder so you can later replace with proper inverse.
        """
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

    def forward(self, x):
        # Decompose input
        branch1_in, branch2_in, (H,V,I) = self.decompose_fn(x)

        # Forward branch 1
        branch1_fea = self.branch1_first_conv(branch1_in)
        branch1_out = self.forward_single_branch(branch1_fea, self.branch1_encoders, self.branch1_bottleneck, self.branch1_decoders, self.branch1_down_layers)
        branch1_out = self.branch1_proj(branch1_out)

        # Forward branch 2
        branch2_fea = self.branch2_first_conv(branch2_in)
        branch2_out = self.forward_single_branch(branch2_fea, self.branch2_encoders, self.branch2_bottleneck, self.branch2_decoders, self.branch2_down_layers)
        branch2_out = self.branch2_proj(branch2_out)

        # Cross-attention
        # We apply cross-attention between the outputs of the two branches
        # Assume same spatial size:
        combined1 = self.cross_attention(branch1_out, branch2_out)
        combined2 = self.cross_attention(branch2_out, branch1_out)
        # Fuse the two combined features
        fused = torch.cat([combined1, combined2], dim=1)

        # Refinement
        out = self.refinement(fused)
        out = self.last_act(out)

        # In this example, we recompose from H,V,I if desired.
        # But we ended up producing RGB already from refinement.
        # If you want to strictly follow the decomposition and recompose after processing the separate components:
        # You can run a recompose function if needed.
        # For demonstration, we will assume refinement outputs final RGB.
        
        return [x, out]
