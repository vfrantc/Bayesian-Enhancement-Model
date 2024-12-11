import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# Re-using LayerNormalization, MSEFBlock, MultiHeadSelfAttention, SEBlock from your snippet

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: B,C,H,W -> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        self._init_weights()

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.pool(x).view(B, C)
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.view(B, C, 1, 1)
        return x * y

    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)


class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se_attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out

    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads

        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        self._init_weights()

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        # x: B,C,H,W
        B, C, H, W = x.size()
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        query = self.split_heads(self.query_dense(x_flat), B)
        key = self.split_heads(self.key_dense(x_flat), B)
        value = self.split_heads(self.value_dense(x_flat), B)

        attention_weights = F.softmax((query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = attention_weights @ value
        attention = attention.permute(0, 2, 1, 3).reshape(B, H, W, C)
        attention = self.combine_heads(attention)
        return attention.permute(0, 3, 1, 2)

    def _init_weights(self):
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias, 0)
        init.constant_(self.key_dense.bias, 0)
        init.constant_(self.value_dense.bias, 0)
        init.constant_(self.combine_heads.bias, 0)


# A more compact model:
# Steps:
# 1. Decompose input into Q1/Q2
# 2. Embed the 8-ch Q-input into some dimension
# 3. Split into two branches: one goes through MSEF (local), another through MHSA (global)
# 4. Concatenate results and do a small fusion
# 5. Output Q1/Q2

class CompactTwoBranchModel(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4):
        super(CompactTwoBranchModel, self).__init__()
        # Embedding
        self.embed = nn.Conv2d(8, embed_dim, kernel_size=3, padding=1)
        init.kaiming_uniform_(self.embed.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.embed.bias is not None:
            init.constant_(self.embed.bias, 0)

        # Two branches:
        # Branch A: Local refinement (MSEFBlock)
        self.msef_branch = MSEFBlock(embed_dim)
        # Branch B: Global reasoning (MHSA)
        self.attn_branch = MultiHeadSelfAttention(embed_size=embed_dim, num_heads=num_heads)

        # Fusion after branches: concat along channels and fuse
        self.fuse = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        init.kaiming_uniform_(self.fuse.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.fuse.bias is not None:
            init.constant_(self.fuse.bias, 0)

        # Output back to 8 channels
        self.out_conv = nn.Conv2d(embed_dim, 8, kernel_size=3, padding=1)
        init.kaiming_uniform_(self.out_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.out_conv.bias is not None:
            init.constant_(self.out_conv.bias, 0)

    def forward(self, inp_img):
        # Decompose into Q1 and Q2
        eps = 1e-8
        R, G, B = torch.split(inp_img, 1, dim=1)
        rgbMax = torch.max(inp_img, dim=1, keepdim=True)[0] + eps

        q1_r = torch.zeros_like(rgbMax)
        q1_i = R / rgbMax
        q1_j = G / rgbMax
        q1_k = B / rgbMax

        q2_r = torch.zeros_like(rgbMax)
        q2_i = R
        q2_j = G
        q2_k = B

        q_input = torch.cat([q1_r, q2_r,
                             q1_i, q2_i,
                             q1_j, q2_j,
                             q1_k, q2_k], dim=1)  # B x 8 x H x W

        x = self.embed(q_input)

        # Branching:
        branch_a = self.msef_branch(x)  # local features
        branch_b = self.attn_branch(x)  # global features

        # Fuse branches:
        combined = torch.cat([branch_a, branch_b], dim=1)
        fused = self.fuse(combined)

        out = self.out_conv(fused)
        # Split back into Q1/Q2
        q1 = out[:, [0, 2, 4, 6], :, :]
        q2 = out[:, [1, 3, 5, 7], :, :]
        return q1, q2

# Example Usage:
# model = CompactTwoBranchModel(embed_dim=32, num_heads=4)
# inp = torch.randn(1, 3, 64, 64)
# q1, q2 = model(inp)
# print(q1.shape, q2.shape)
