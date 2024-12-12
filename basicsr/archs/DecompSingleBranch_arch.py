import sys
import os
import math
import warnings
import torch
import torch.nn as nn
from functools import partial
from basicsr.archs.arch_util import SAM
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.QD.quaternion import hamilton_product

try:
    from vmamba.models.vmamba import VSSBlock, LayerNorm2d
except:
    from ..vmamba.models.vmamba import VSSBlock, LayerNorm2d

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
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

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
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
class DecompSingleBranch(nn.Module):

    def __init__(
        self,
        in_channels=6,          # 3 for image + 3 for conditioning
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
        decomp_model='model1'
    ):
        super(DecompSingleBranch, self).__init__()
        self.stage = stage
        self.num_levels = len(num_blocks)
        if isinstance(d_state, int):
            d_state = [d_state]*self.num_levels

        # Load decomposition model
        if decomp_model == 'model1':
            from basicsr.QD.model1 import Decomp
            model_file = 'basicsr/QD/checkpoints/model1_999.pth'
        elif decomp_model == 'model2':
            from basicsr.QD.model2 import Decomp
            model_file = 'basicsr/QD/checkpoints/model2_999.pth'
        elif decomp_model == 'model3':
            from basicsr.QD.model3 import Decomp
            model_file = 'basicsr/QD/checkpoints/model3_999.pth'
        elif decomp_model == 'model4':
            from basicsr.QD.model4 import Decomp
            model_file = 'basicsr/QD/checkpoints/model4_999.pth'
        else:
            raise ValueError(f"Unknown decomp_model: {decomp_model}")

        self.decomp = Decomp(use_wavelets=True)
        self.decomp.load_state_dict(torch.load(model_file)['model_state_dict'])
        self.decomp.eval()
        for param in self.decomp.parameters():
            param.requires_grad = False

        # We have Q1 and Q2 (8 channels total) from the decomposition.
        # Additionally, we have 3 conditioning channels. Total = 8 + 3 = 11 channels.
        self.conditioning_channels = 3
        in_channels_unet = 8 + self.conditioning_channels
        out_channels_unet = 8

        if use_pixelshuffle:
            down_layer = conv_down
            up_layer = deconv_up
        else:
            down_layer = conv_down
            up_layer = deconv_up

        self.first_conv = nn.Conv2d(in_channels_unet, n_feat, 3, 1, 1, bias=True)
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

        self.proj = nn.Conv2d(n_feat, out_channels_unet, 3, 1, 1, bias=True)
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
        # x: Nx6xHxW, first 3 are image, last 3 are conditioning
        img = x[:, :3, :, :]   # Nx3xHxW
        cond = x[:, 3:, :, :]  # Nx3xHxW

        # Decompose image into Q1,Q2
        with torch.no_grad():
            Q1, Q2 = self.decomp(img)  # Q1,Q2: Nx4xHxW

        # Concatenate Q1, Q2 and conditioning: Nx(8+3)xHxW = Nx11xHxW
        fea = torch.cat([Q1, Q2, cond], dim=1)

        # Pass through UNet
        fea = self.first_conv(fea)

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

        out = self.proj(curr_feat)
        out = self.last_act(out)  # out: Nx8xHxW

        # Split output into Q1_out and Q2_out
        Q1_out = out[:, :4, :, :]
        Q2_out = out[:, 4:, :, :]

        # Perform Hamilton product
        out_quat = hamilton_product(Q1_out, Q2_out)  # Nx4xHxW
        # Ignore the real part
        final_out = out_quat[:, 1:, :, :]  # Nx3xHxW

        return [x, final_out]

def build_model():
    return DecompSingleBranch(
        stage=1,
        n_feat=40,
        num_blocks=[2, 2, 2],
        d_state=[1, 1, 1],
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        use_pixelshuffle=True,
        decomp_model='model1'
    )
