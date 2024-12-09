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
class VMUNet(nn.Module):
    """
    A simplified VMUNet model with classical U-Net structure, using VSSBlock as the basic building block.
    No Bayesian layers, no complicated layers, just encoders and decoders with VSSBlock.
    This class can be instantiated with the same parameters as the original complex model.

    Args:
        in_channels (int): input channel number
        out_channels (int): output channel number
        n_feat (int): channel number of intermediate features
        stage (int): number of stages (outer loops). Usually 1 for a single forward pass U-Net.
        num_blocks (list): each element defines the number of VSS blocks in a scale level
        d_state (int or list): dimension(s) of the hidden state in SSM
        ssm_ratio (int): expansion ratio of SSM in VSSBlock
        mlp_ratio (float): expansion ratio of MLP in VSSBlock
        mlp_type (str): MLP type used in VSSBlock
        use_pixelshuffle (bool): whether to use pixelshuffle-based up/down sampling.
                                 If False, uses simple conv down and deconv up.
        drop_path (float): ratio of drop_path (not used here, but kept for compatibility)
        use_illu (bool): not used here, just kept for compatibility
        sam (bool): not used here, no SAM blocks included
        last_act (str): if not None, apply activation after final projection. (e.g. "relu", "softmax")
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
        super(VMUNet, self).__init__()
        self.stage = stage
        self.num_levels = len(num_blocks)
        if isinstance(d_state, int):
            d_state = [d_state]*self.num_levels
        # Downsample and upsample layers
        if use_pixelshuffle:
            # Pixel shuffle down/up not requested to be simplified, 
            # but user wants classical structure. Let's just do classical conv/down and deconv/up.
            # The user said "just classical structure", so let's not complicate with pixelshuffle.
            down_layer = conv_down
            up_layer = deconv_up
        else:
            down_layer = conv_down
            up_layer = deconv_up

        self.first_conv = nn.Conv2d(in_channels, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(
            self.first_conv.weight, mode="fan_out", nonlinearity="linear"
        )
        if self.first_conv.bias is not None:
            nn.init.zeros_(self.first_conv.bias)

        # Encoder
        self.encoders = nn.ModuleList()
        curr_dim = n_feat
        for i in range(self.num_levels - 1):
            self.encoders.append(self._make_level(curr_dim, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type))
            curr_dim *= 2

        # Bottleneck
        self.bottleneck = self._make_level(curr_dim, num_blocks[-1], d_state[-1], ssm_ratio, mlp_ratio, mlp_type)

        # Decoder
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

        # For U-Net structure, we need to store intermediate encoder results for skip connections
        self.down_layers = nn.ModuleList([down_layer(n_feat * (2 ** i)) for i in range(self.num_levels - 1)])

        # Drop path is not used here, but we keep the signature
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
        """Create a level of VSSBlocks."""
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
        # We do not use mask or bayesian features here, just a classical forward pass.
        out_list = [x]
        fea = self.first_conv(x)

        # Encoding
        skip_connections = []
        curr_feat = fea
        for i in range(self.num_levels - 1):
            curr_feat = self.encoders[i](curr_feat)
            skip_connections.append(curr_feat)
            curr_feat = self.down_layers[i](curr_feat)

        # Bottleneck
        curr_feat = self.bottleneck(curr_feat)

        # Decoding
        for i, dec in enumerate(self.decoders):
            skip = skip_connections[self.num_levels - 2 - i]
            curr_feat = dec['up'](curr_feat)
            curr_feat = dec['fuse'](torch.cat([curr_feat, skip], dim=1))
            curr_feat = dec['block'](curr_feat)

        out = self.proj(curr_feat)
        out = self.last_act(out)
        out_list.append(out)
        return out_list


def build_model():
    return VMUNet(
        stage=1,
        n_feat=40,
        num_blocks=[2, 2, 2],
        d_state=[1, 1, 1],
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        use_pixelshuffle=True,
    )