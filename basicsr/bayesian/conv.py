import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from itertools import repeat
from torch.nn import Parameter
from .base_layer import BaseLayer_
import math

class Conv2dReparameterization(BaseLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 sigma_init=0.05,
                 decay=0.9998):
        """
        Args:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super(Conv2dReparameterization, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        self.deterministic = False # set to True to get deterministic output
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.decay = decay
        self.sigma_init = sigma_init
        self.step = 0

        kernel_size = get_kernel_size(kernel_size, 2)

        self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        self.rho_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        self.register_buffer('eps_weight',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]), persistent=False)
        self.register_buffer('prior_mu_weight',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]), persistent=False)
        self.register_buffer('prior_rho_weight',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]), persistent=False)

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_mu_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_rho_bias', torch.Tensor(out_channels), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        rho_init = math.log(math.expm1(abs(self.sigma_init)) + 1e-20)
        nn.init.kaiming_normal_(self.mu_weight, mode='fan_in', nonlinearity='leaky_relu')
        self.rho_weight.data.fill_(rho_init)
        self.prior_mu_weight.data.copy_(self.mu_weight.data)
        self.prior_rho_weight.data.copy_(self.rho_weight.data)
        if self.bias:
            self.mu_bias.data.fill_(0)
            self.rho_bias.data.fill_(rho_init)
            self.prior_mu_bias.data.copy_(self.mu_bias.data)
            self.prior_rho_bias.data.copy_(self.rho_bias.data)

    def kl_loss(self):
        kl = self.kl_div(self.mu_weight, self.sigma_weight, self.prior_mu_weight, self.prior_sigma_weight)
        if self.bias:
            kl += self.kl_div(self.mu_bias, self.sigma_bias, self.prior_mu_bias, self.prior_sigma_bias)
        return kl

    def _forward_uncertain(self, input):
        if self.training:
            with torch.no_grad():
                _decay = min(self.decay, (1 + self.step) / (10 + self.step)) # threshold EMA
                # _decay = self.decay
                self.prior_mu_weight = _decay * self.prior_mu_weight + (1 - _decay) * self.mu_weight
                self.prior_rho_weight = _decay * self.prior_rho_weight + (1 - _decay) * self.rho_weight
                self.prior_sigma_weight = torch.log1p(torch.exp(self.prior_rho_weight))

                if self.bias:
                    self.prior_mu_bias = _decay * self.prior_mu_bias + (1 - _decay) * self.mu_bias
                    self.prior_rho_bias = _decay * self.prior_rho_bias + (1 - _decay) * self.rho_bias
                    self.prior_sigma_bias = torch.log1p(torch.exp(self.prior_rho_bias))
            self.step += 1

        self.sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        weight = self.mu_weight + self.sigma_weight * self.eps_weight.data.normal_()
        if self.bias:
            self.sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + self.sigma_bias * self.eps_bias.data.normal_()
        else:
            bias = None

        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        return out

    def _forward_det(self, input):

        weight = self.mu_weight
        if self.bias:
            bias = self.mu_bias
        else:
            bias = None

        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        return out


def get_kernel_size(x, n):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

