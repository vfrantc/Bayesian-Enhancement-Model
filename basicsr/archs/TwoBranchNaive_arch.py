# # psnr: 27.9652         # ssim: 0.8773

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


@ARCH_REGISTRY.register()
class NaiveVMUNetTwoBranch(nn.Module):
    """
    A two-branch VMUNet model that mirrors the original VMUNet structure twice, and fuses
    the final outputs from both branches before returning. The signature and returned
    format remain identical to the single-branch VMUNet, i.e., return [input, output].

    Args are the same as VMUNet, so it can be plugged in directly.
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
        drop_path=0.0,
        use_illu=False,
        sam=False,
        last_act=None,
    ):
        super(NaiveVMUNetTwoBranch, self).__init__()
        self.stage = stage
        self.num_levels = len(num_blocks)
        if isinstance(d_state, int):
            d_state = [d_state]*self.num_levels

        if use_pixelshuffle:
            down_layer = conv_down
            up_layer = deconv_up
        else:
            down_layer = conv_down
            up_layer = deconv_up

        # ===================== Branch 1 ======================
        self.first_conv = nn.Conv2d(in_channels, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(
            self.first_conv.weight, mode="fan_out", nonlinearity="linear"
        )
        if self.first_conv.bias is not None:
            nn.init.zeros_(self.first_conv.bias)

        self.encoders = nn.ModuleList()
        curr_dim = n_feat
        for i in range(self.num_levels - 1):
            self.encoders.append(self._make_level(curr_dim, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type))
            curr_dim *= 2

        self.bottleneck = self._make_level(curr_dim, num_blocks[-1], d_state[-1], ssm_ratio, mlp_ratio, mlp_type)

        self.decoders = nn.ModuleList()
        for i in range(self.num_levels - 2, -1, -1):
            self.decoders.append(nn.ModuleDict({
                'up': up_layer(curr_dim),
                'fuse': nn.Conv2d(curr_dim, curr_dim // 2, 1, 1, bias=False),
                'block': self._make_level(curr_dim // 2, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type)
            }))
            curr_dim //= 2

        self.proj = nn.Conv2d(n_feat, out_channels, 3, 1, 1, bias=True)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        if last_act is None:
            self.last_act = nn.Identity()
        elif last_act == "relu":
            self.last_act = nn.ReLU()
        elif last_act == "softmax":
            self.last_act = nn.Softmax(dim=1)
        else:
            raise NotImplementedError

        self.down_layers = nn.ModuleList([down_layer(n_feat * (2 ** i)) for i in range(self.num_levels - 1)])

        # ===================== Branch 2 ======================
        self.first_conv2 = nn.Conv2d(in_channels, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(
            self.first_conv2.weight, mode="fan_out", nonlinearity="linear"
        )
        if self.first_conv2.bias is not None:
            nn.init.zeros_(self.first_conv2.bias)

        self.encoders2 = nn.ModuleList()
        curr_dim = n_feat
        for i in range(self.num_levels - 1):
            self.encoders2.append(self._make_level(curr_dim, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type))
            curr_dim *= 2

        self.bottleneck2 = self._make_level(curr_dim, num_blocks[-1], d_state[-1], ssm_ratio, mlp_ratio, mlp_type)

        self.decoders2 = nn.ModuleList()
        curr_dim = n_feat * (2 ** (self.num_levels - 1))
        for i in range(self.num_levels - 2, -1, -1):
            self.decoders2.append(nn.ModuleDict({
                'up': up_layer(curr_dim),
                'fuse': nn.Conv2d(curr_dim, curr_dim // 2, 1, 1, bias=False),
                'block': self._make_level(curr_dim // 2, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type)
            }))
            curr_dim //= 2

        self.proj2 = nn.Conv2d(n_feat, out_channels, 3, 1, 1, bias=True)
        if self.proj2.bias is not None:
            nn.init.zeros_(self.proj2.bias)

        # The second branch shares the same type of final activation
        if isinstance(self.last_act, nn.Identity):
            self.last_act2 = nn.Identity()
        else:
            # If it's a stateful module, we assume it can be recreated. 
            # If not, one might directly reuse self.last_act (assuming it's stateless).
            # For safety, just reuse the same module reference:
            self.last_act2 = self.last_act

        self.down_layers2 = nn.ModuleList([down_layer(n_feat * (2 ** i)) for i in range(self.num_levels - 1)])

        # Drop path is not used
        self.drop_path = nn.Identity()

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

    def forward(self, x, mask=None):
        # ========== Branch 1 Forward ==========
        fea = self.first_conv(x)
        skip_connections = []
        curr_feat = fea
        for i in range(self.num_levels - 1):
            curr_feat = self.encoders[i](curr_feat)
            skip_connections.append(curr_feat)
            curr_feat = self.down_layers[i](curr_feat)
        curr_feat = self.bottleneck(curr_feat)
        for i, dec in enumerate(self.decoders):
            skip = skip_connections[self.num_levels - 2 - i]
            curr_feat = dec['up'](curr_feat)
            curr_feat = dec['fuse'](torch.cat([curr_feat, skip], dim=1))
            curr_feat = dec['block'](curr_feat)
        out_1 = self.proj(curr_feat)
        out_1 = self.last_act(out_1)

        # ========== Branch 2 Forward ==========
        fea2 = self.first_conv2(x)
        skip_connections2 = []
        curr_feat2 = fea2
        for i in range(self.num_levels - 1):
            curr_feat2 = self.encoders2[i](curr_feat2)
            skip_connections2.append(curr_feat2)
            curr_feat2 = self.down_layers2[i](curr_feat2)
        curr_feat2 = self.bottleneck2(curr_feat2)
        for i, dec2 in enumerate(self.decoders2):
            skip2 = skip_connections2[self.num_levels - 2 - i]
            curr_feat2 = dec2['up'](curr_feat2)
            curr_feat2 = dec2['fuse'](torch.cat([curr_feat2, skip2], dim=1))
            curr_feat2 = dec2['block'](curr_feat2)
        out_2 = self.proj2(curr_feat2)
        out_2 = self.last_act2(out_2)

        # ========== Fusion of Outputs ==========
        # We fuse the two outputs by averaging them, returning the same format: [input, fused_output]
        fused_out = (out_1 + out_2) / 2.0

        # Return same format as original: [x, out]
        return [x, fused_out]


def build_model():
    return VMUNetTwoBranch(
        stage=1,
        n_feat=40,
        num_blocks=[2, 2, 2],
        d_state=[1, 1, 1],
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        use_pixelshuffle=True,
    )
