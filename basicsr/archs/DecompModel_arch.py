import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import math
import warnings
from basicsr.utils.registry import ARCH_REGISTRY
try:
    from ..vmamba.models.vmamba import VSSBlock, LayerNorm2d
except:
    from vmamba.models.vmamba import VSSBlock, LayerNorm2d

from basicsr.QD.quaternion import hamilton_product

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

class CrossFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super(CrossFusionBlock, self).__init__()
        self.transform = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.gate = nn.Parameter(torch.ones(1, in_channels, 1, 1))

    def forward(self, x_src, x_tgt):
        # fused: x_tgt + gate * transform(x_src)
        fused = x_tgt + self.gate * self.transform(x_src)
        return fused

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_attn = torch.cat([avg_out, max_out], dim=1)
        x_attn = self.conv(x_attn)
        return x * self.sigmoid(x_attn)

@ARCH_REGISTRY.register()
class DecompDualBranch(nn.Module):
    """
    A two-branch VMUNet model with a single cross-level fusion at the deepest encoder stage.
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
        decomp_model='model1'
    ):
        super(DecompDualBranch, self).__init__()

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
        self.decomp = Decomp(use_wavelets=True)
        self.decomp.load_state_dict(torch.load(model_file)['model_state_dict'])
        self.decomp.eval()
        for param in self.decomp.parameters():
            param.requires_grad = False

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
        self.first_conv = nn.Conv2d(4, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(
            self.first_conv.weight, mode="fan_out", nonlinearity="linear"
        )
        if self.first_conv.bias is not None:
            nn.init.zeros_(self.first_conv.bias)

        self.encoders = nn.ModuleList()
        curr_dim = n_feat
        for i in range(self.num_levels - 1):
            self.encoders.append(
                self._make_level(curr_dim, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type)
            )
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

        self.proj = nn.Conv2d(n_feat, 4, 3, 1, 1, bias=True)
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
        self.first_conv2 = nn.Conv2d(4, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(
            self.first_conv2.weight, mode="fan_out", nonlinearity="linear"
        )
        if self.first_conv2.bias is not None:
            nn.init.zeros_(self.first_conv2.bias)

        self.encoders2 = nn.ModuleList()
        curr_dim = n_feat
        for i in range(self.num_levels - 1):
            self.encoders2.append(
                self._make_level(curr_dim, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type)
            )
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

        self.proj2 = nn.Conv2d(n_feat, 4, 3, 1, 1, bias=True)
        if self.proj2.bias is not None:
            nn.init.zeros_(self.proj2.bias)

        # Same final activation for branch 2
        if isinstance(self.last_act, nn.Identity):
            self.last_act2 = nn.Identity()
        else:
            self.last_act2 = self.last_act

        self.down_layers2 = nn.ModuleList([down_layer(n_feat * (2 ** i)) for i in range(self.num_levels - 1)])

        # Drop path not used
        self.drop_path = nn.Identity()

        # Define a single cross-fusion at the deepest level
        bottleneck_dim = n_feat * (2 ** (self.num_levels - 1))
        self.cross_fusion_12 = CrossFusionBlock(bottleneck_dim)
        self.cross_fusion_21 = CrossFusionBlock(bottleneck_dim)

        # SE block and spatial attention
        self.bottleneck_se = SEBlock(bottleneck_dim)
        self.bottleneck_se2 = SEBlock(bottleneck_dim)

        self.spatial_attention = SpatialAttention()
        self.spatial_attention2 = SpatialAttention()

        # self.fusion = nn.Sequential(
        #     nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        # )

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
        # take only first 3 channels
        x1, x2 = self.decomp(x[:, :3, :, :])
        # ===== Branch 1 encoder =====
        fea = self.first_conv(x1)
        skip_connections = []
        curr_feat = fea
        for i in range(self.num_levels - 1):
            curr_feat = self.encoders[i](curr_feat)
            skip_connections.append(curr_feat)
            curr_feat = self.down_layers[i](curr_feat)

        # ===== Branch 2 encoder =====
        fea2 = self.first_conv2(x2)
        skip_connections2 = []
        curr_feat2 = fea2
        for i in range(self.num_levels - 1):
            curr_feat2 = self.encoders2[i](curr_feat2)
            skip_connections2.append(curr_feat2)
            curr_feat2 = self.down_layers2[i](curr_feat2)

        # ===== Single cross-level fusion at the deepest level =====
        curr_feat2 = self.cross_fusion_12(curr_feat, curr_feat2)
        curr_feat = self.cross_fusion_21(curr_feat2, curr_feat)

        # ===== Bottleneck and attention =====
        curr_feat = self.bottleneck(curr_feat)
        curr_feat = self.bottleneck_se(curr_feat)
        curr_feat = self.spatial_attention(curr_feat)

        curr_feat2 = self.bottleneck2(curr_feat2)
        curr_feat2 = self.bottleneck_se2(curr_feat2)
        curr_feat2 = self.spatial_attention2(curr_feat2)

        # ===== Branch 1 decoder =====
        for i, dec in enumerate(self.decoders):
            skip = skip_connections[self.num_levels - 2 - i]
            curr_feat = dec['up'](curr_feat)
            curr_feat = dec['fuse'](torch.cat([curr_feat, skip], dim=1))
            curr_feat = dec['block'](curr_feat)
        out_1 = self.proj(curr_feat)
        out_1 = self.last_act(out_1)

        # ===== Branch 2 decoder =====
        for i, dec2 in enumerate(self.decoders2):
            skip2 = skip_connections2[self.num_levels - 2 - i]
            curr_feat2 = dec2['up'](curr_feat2)
            curr_feat2 = dec2['fuse'](torch.cat([curr_feat2, skip2], dim=1))
            curr_feat2 = dec2['block'](curr_feat2)
        out_2 = self.proj2(curr_feat2)
        out_2 = self.last_act2(out_2)

        # ===== Final fusion of two branch outputs =====
        fused_out = hamilton_product(out_1, out_2)
        return [x, fused_out[:, 1:, :, :]]


def build_model():
    return FusedTunedModel(
        stage=1,
        n_feat=40,
        num_blocks=[2, 2, 2],
        d_state=[1, 1, 1],
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        use_pixelshuffle=True,
    )
