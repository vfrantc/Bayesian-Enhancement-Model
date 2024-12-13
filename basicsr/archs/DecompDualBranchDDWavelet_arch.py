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
from basicsr.QD.model4 import DWT, IWT  # Ensure this is correct path

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


### Custom MyDecomp class ###
# This class will inherit from the chosen Decomp model (model1, model2, model3, or model4)
# and override forward to return Q1_w, Q2_w before IWT and smoothing.
# We do not define smoothing layers or use them. We load weights with strict=False.
def create_my_decomp(decomp_model):
    if decomp_model == 'model1':
        from basicsr.QD.model1 import Decomp as BaseDecomp
        model_file = 'basicsr/QD/checkpoints/model1_999.pth'
    elif decomp_model == 'model2':
        from basicsr.QD.model2 import Decomp as BaseDecomp
        model_file = 'basicsr/QD/checkpoints/model2_999.pth'
    elif decomp_model == 'model3':
        from basicsr.QD.model3 import Decomp as BaseDecomp
        model_file = 'basicsr/QD/checkpoints/model3_999.pth'
    elif decomp_model == 'model4':
        from basicsr.QD.model4 import Decomp as BaseDecomp
        model_file = 'basicsr/QD/checkpoints/model4_999.pth'
    else:
        raise ValueError(f"Unknown decomp_model: {decomp_model}")

    class MyDecomp(BaseDecomp):
        def __init__(self, **kwargs):
            super(MyDecomp, self).__init__(**kwargs)
            # Remove smoothing layers if defined in base class
            if hasattr(self, 'smooth_q1'):
                del self.smooth_q1
            if hasattr(self, 'smooth_q2'):
                del self.smooth_q2

        def forward(self, inp_img):
            # This forward is a modified version that stops before IWT and smoothing.
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
                input_tensor = self.dwt(input_tensor)  # Nx8x(H/2)x(W/2)

            feat = self.conv_in(input_tensor)
            q1_feat = self.branch_q1(feat) + feat
            q2_feat = self.branch_q2(feat) + feat
            q1_feat, q2_feat = self.cross_attn(q1_feat, q2_feat)

            fused_feat = torch.cat([q1_feat, q2_feat], dim=1)
            fused_feat = self.fuse(fused_feat)
            out = self.conv_out(fused_feat)

            sharp_map = self.sharpening(out)
            out = out + sharp_map
            # Do NOT perform IWT or smoothing here.
            # out shape: Nx8x(H/2)x(W/2) in wavelet domain

            # Split Q1 and Q2
            q1_w = out[:, [0, 2, 4, 6], :, :]
            q2_w = out[:, [1, 3, 5, 7], :, :]

            return q1_w, q2_w  # wavelet domain Q1/Q2

    # Instantiate and load weights
    my_decomp = MyDecomp(use_wavelets=True)
    state_dict = torch.load(model_file)['model_state_dict']
    # Load with strict=False to ignore smoothing layers
    my_decomp.load_state_dict(state_dict, strict=False)
    my_decomp.eval()
    for p in my_decomp.parameters():
        p.requires_grad = False

    return my_decomp


@ARCH_REGISTRY.register()
class DecompDualBranchDDWavelet(nn.Module):
    """
    DecompDualBranchDDWavelet:
    Similar to previous but we now rely on a modified decomposition model (MyDecomp) that
    already returns Q1 and Q2 in wavelet domain. We just concatenate image and condition Q1/Q2,
    run them through dual-branch encoders/decoders in the wavelet domain, then apply IWT
    and the Hamilton product at the end.
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
        super(DecompDualBranchDDWavelet, self).__init__()
        self.stage = stage
        self.num_levels = len(num_blocks)
        if isinstance(d_state, int):
            d_state = [d_state]*self.num_levels

        # Use the custom MyDecomp
        self.decomp = create_my_decomp(decomp_model)

        # Wavelet domain Q1/Q2 each have 4 channels after decomposition (Q1_w Nx4, Q2_w Nx4).
        # After concatenating img and cond: Nx8 for Q1 and Nx8 for Q2.
        in_channels_branch = 8
        # We want to output Nx8 again (4 Q1 + 4 Q2 for each branch final),
        # but after decoding we produce Nx8 and then split into Q1_out_w Nx4 and Q2_out_w Nx4.
        # Actually, the model outputs Q1_out_w Nx4 and Q2_out_w Nx4 at the end. So final projection is Nx4.
        out_channels_branch = 4

        if use_pixelshuffle:
            down_layer = conv_down
            up_layer = deconv_up
        else:
            down_layer = conv_down
            up_layer = deconv_up

        # --- Encoder for Q1 ---
        self.first_conv_Q1 = nn.Conv2d(in_channels_branch, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(self.first_conv_Q1.weight, mode="fan_out", nonlinearity="linear")
        if self.first_conv_Q1.bias is not None:
            nn.init.zeros_(self.first_conv_Q1.bias)

        self.encoders_Q1 = nn.ModuleList()
        curr_dim = n_feat
        for i in range(self.num_levels - 1):
            self.encoders_Q1.append(self._make_level(curr_dim, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type))
            curr_dim *= 2
        self.down_layers_Q1 = nn.ModuleList([down_layer(n_feat * (2 ** i)) for i in range(self.num_levels - 1)])

        # --- Encoder for Q2 ---
        self.first_conv_Q2 = nn.Conv2d(in_channels_branch, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(self.first_conv_Q2.weight, mode="fan_out", nonlinearity="linear")
        if self.first_conv_Q2.bias is not None:
            nn.init.zeros_(self.first_conv_Q2.bias)

        self.encoders_Q2 = nn.ModuleList()
        curr_dim = n_feat
        for i in range(self.num_levels - 1):
            self.encoders_Q2.append(self._make_level(curr_dim, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type))
            curr_dim *= 2
        self.down_layers_Q2 = nn.ModuleList([down_layer(n_feat * (2 ** i)) for i in range(self.num_levels - 1)])

        # --- Bottleneck ---
        fused_dim = curr_dim * 2
        self.bottleneck_fuse = nn.Conv2d(fused_dim, curr_dim, 1, 1, bias=False)
        self.bottleneck_block = self._make_level(curr_dim, num_blocks[-1], d_state[-1], ssm_ratio, mlp_ratio, mlp_type)

        self.bottleneck_to_Q1 = nn.Conv2d(curr_dim, curr_dim, 1, 1, bias=False)
        self.bottleneck_to_Q2 = nn.Conv2d(curr_dim, curr_dim, 1, 1, bias=False)

        # --- Decoder for Q1 ---
        dec_curr_dim = curr_dim
        self.decoders_Q1 = nn.ModuleList()
        for i in range(self.num_levels - 2, -1, -1):
            self.decoders_Q1.append(nn.ModuleDict({
                'up': up_layer(dec_curr_dim),
                'fuse': nn.Conv2d(dec_curr_dim, dec_curr_dim // 2, 1, 1, bias=False),
                'block': self._make_level(dec_curr_dim // 2, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type)
            }))
            dec_curr_dim //= 2
        self.proj_Q1 = nn.Conv2d(n_feat, out_channels_branch, 3, 1, 1, bias=True)
        if self.proj_Q1.bias is not None:
            nn.init.zeros_(self.proj_Q1.bias)

        # --- Decoder for Q2 ---
        dec_curr_dim = curr_dim
        self.decoders_Q2 = nn.ModuleList()
        for i in range(self.num_levels - 2, -1, -1):
            self.decoders_Q2.append(nn.ModuleDict({
                'up': up_layer(dec_curr_dim),
                'fuse': nn.Conv2d(dec_curr_dim, dec_curr_dim // 2, 1, 1, bias=False),
                'block': self._make_level(dec_curr_dim // 2, num_blocks[i], d_state[i], ssm_ratio, mlp_ratio, mlp_type)
            }))
            dec_curr_dim //= 2
        self.proj_Q2 = nn.Conv2d(n_feat, out_channels_branch, 3, 1, 1, bias=True)
        if self.proj_Q2.bias is not None:
            nn.init.zeros_(self.proj_Q2.bias)

        if last_act is None:
            self.last_act = nn.Identity()
        elif last_act == "relu":
            self.last_act = nn.ReLU()
        elif last_act == "softmax":
            self.last_act = nn.Softmax(dim=1)
        else:
            raise NotImplementedError

        self.iwt = IWT()  # We apply IWT at the end

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
        # x: Nx6xHxW (first 3: image, last 3: conditioning)
        img = x[:, 0:3, :, :]
        cond = x[:, 3:6, :, :]

        # Decompose into Q1_w/Q2_w in wavelet domain using MyDecomp
        with torch.no_grad():
            Q1_img_w, Q2_img_w = self.decomp(img)     # Nx4x(H/2)x(W/2)
            Q1_cond_w, Q2_cond_w = self.decomp(cond)  # Nx4x(H/2)x(W/2)

        # Concatenate along channel dimension
        # Q1: Nx8x(H/2)x(W/2), Q2: Nx8x(H/2)x(W/2)
        Q1 = torch.cat([Q1_img_w, Q1_cond_w], dim=1)
        Q2 = torch.cat([Q2_img_w, Q2_cond_w], dim=1)

        # --- Encoder forward Q1 ---
        fea_Q1 = self.first_conv_Q1(Q1)
        skip_connections_Q1 = []
        curr_feat_Q1 = fea_Q1
        for i in range(self.num_levels - 1):
            curr_feat_Q1 = self.encoders_Q1[i](curr_feat_Q1)
            skip_connections_Q1.append(curr_feat_Q1)
            curr_feat_Q1 = self.down_layers_Q1[i](curr_feat_Q1)

        # --- Encoder forward Q2 ---
        fea_Q2 = self.first_conv_Q2(Q2)
        skip_connections_Q2 = []
        curr_feat_Q2 = fea_Q2
        for i in range(self.num_levels - 1):
            curr_feat_Q2 = self.encoders_Q2[i](curr_feat_Q2)
            skip_connections_Q2.append(curr_feat_Q2)
            curr_feat_Q2 = self.down_layers_Q2[i](curr_feat_Q2)

        # --- Bottleneck fusion ---
        fused = torch.cat([curr_feat_Q1, curr_feat_Q2], dim=1)
        fused = self.bottleneck_fuse(fused)
        fused = self.bottleneck_block(fused)

        dec_start_Q1 = self.bottleneck_to_Q1(fused)
        dec_start_Q2 = self.bottleneck_to_Q2(fused)

        # --- Decoder forward Q1 ---
        curr_feat = dec_start_Q1
        for i, dec in enumerate(self.decoders_Q1):
            skip = skip_connections_Q1[self.num_levels - 2 - i]
            curr_feat = dec['up'](curr_feat)
            curr_feat = dec['fuse'](torch.cat([curr_feat, skip], dim=1))
            curr_feat = dec['block'](curr_feat)
        Q1_out_w = self.proj_Q1(curr_feat)  # Nx4x(H/2)x(W/2)
        Q1_out_w = self.last_act(Q1_out_w)

        # --- Decoder forward Q2 ---
        curr_feat = dec_start_Q2
        for i, dec in enumerate(self.decoders_Q2):
            skip = skip_connections_Q2[self.num_levels - 2 - i]
            curr_feat = dec['up'](curr_feat)
            curr_feat = dec['fuse'](torch.cat([curr_feat, skip], dim=1))
            curr_feat = dec['block'](curr_feat)
        Q2_out_w = self.proj_Q2(curr_feat)  # Nx4x(H/2)x(W/2)
        Q2_out_w = self.last_act(Q2_out_w)

        # Apply IWT to go back to spatial domain Nx4xHxW
        Q1_out = self.iwt(Q1_out_w)
        Q2_out = self.iwt(Q2_out_w)

        # --- Hamilton product ---
        out_quat = hamilton_product(Q1_out, Q2_out)  # Nx4xHxW
        final_out = out_quat[:, 1:, :, :]  # Nx3xHxW

        # Return x (same 6 channels as input) and final_out (3 channels)
        return [x, final_out]


def build_model():
    return DecompDualBranchDDWavelet(
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
