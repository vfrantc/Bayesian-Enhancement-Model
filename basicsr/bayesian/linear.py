import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from .base_layer import BaseLayer_
import math

class Linear2dReparameterization(BaseLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 sigma_init=0.05,
                 decay=0.9998):
        """
        Args:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(Linear2dReparameterization, self).__init__()

        self.deterministic = False # set to True to get deterministic output
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.decay = decay
        self.sigma_init = sigma_init
        self.step = 0

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight', torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('prior_mu_weight', torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('prior_rho_weight', torch.Tensor(out_features, in_features), persistent=False)
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer('eps_bias', torch.Tensor(out_features), persistent=False)
            self.register_buffer('prior_mu_bias', torch.Tensor(out_features), persistent=False)
            self.register_buffer('prior_rho_bias', torch.Tensor(out_features), persistent=False)


        self.init_parameters()

    def init_parameters(self):
        rho_init = math.log(math.expm1(abs(self.sigma_init)) + 1e-20)
        nn.init.xavier_uniform_(self.mu_weight)
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
                _decay = min(self.decay, (1 + self.step) / (10 + self.step))
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

        out = F.conv2d(input, weight[:, :, None, None], bias)

        return out

    def _forward_det(self, input):

        weight = self.mu_weight
        if self.bias:
            bias = self.mu_bias
        else:
            bias = None

        out = F.conv2d(input, weight[:, :, None, None], bias)

        return out

class LinearReparameterization(BaseLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 sigma_init=0.05,
                 decay=0.9998):
        """
        Args:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(LinearReparameterization, self).__init__()

        self.deterministic = False # set to True to get deterministic output
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.decay = decay
        self.sigma_init = sigma_init
        self.step = 0

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight', torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('prior_mu_weight', torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('prior_rho_weight', torch.Tensor(out_features, in_features), persistent=False)
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer('eps_bias', torch.Tensor(out_features), persistent=False)
            self.register_buffer('prior_mu_bias', torch.Tensor(out_features), persistent=False)
            self.register_buffer('prior_rho_bias', torch.Tensor(out_features), persistent=False)


        self.init_parameters()

    def init_parameters(self):
        rho_init = math.log(math.expm1(abs(self.sigma_init)) + 1e-20)
        nn.init.xavier_uniform_(self.mu_weight)
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
                _decay = min(self.decay, (1 + self.step) / (10 + self.step))
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

        out = F.linear(input, weight, bias)

        return out

    def _forward_det(self, input):

        weight = self.mu_weight
        if self.bias:
            bias = self.mu_bias
        else:
            bias = None

        out = F.linear(input, weight, bias)

        return out