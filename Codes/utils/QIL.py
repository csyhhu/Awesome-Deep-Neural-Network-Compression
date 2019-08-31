"""
Helper functions and module for QIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

# from utils.quantize import Function_STE
from utils.train import progress_bar, accuracy, AverageMeter


class Function_interval_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, p, c, bitW):
        # ctx.save_for_backward(weight)
        n = float(2 ** bitW - 1)
        interval = (c-p) / n
        return torch.round((weight - p) / interval) * interval + p

    @staticmethod
    def backward(ctx, grad_outputs):

        return grad_outputs, None


class Function_STE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, bitW):
        ctx.save_for_backward(weight)

        n = float(2 ** (bitW) - 1)
        # return torch.sign(weight) * torch.round(torch.abs(weight) * n) / n
        return torch.round(weight * n) / n

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, = ctx.saved_tensors
        gate = (torch.abs(weight) <= 1).float()
        grad_inputs = grad_outputs * gate
        return grad_inputs, None


class QIL_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, bitW=1):
        super(QIL_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.bitW = bitW

        # self.pruning_point = nn.Parameter(torch.zeros([])) # c_w - d_W
        self.pruning_point = nn.Parameter(torch.Tensor([0.0]))  # c_w - d_W
        self.clipping_point = nn.Parameter(torch.Tensor([1.0])) # c_w + d_W
        # c_W = 1/2 * (pruning_point + clipping_point), d_w = 1/2 * (clipping_point - pruning_point)
        # alpha_W = 0.5 / d_w, beta_w = -0.5 * c_W / d_W + 0.5
        self.gamma = nn.Parameter(torch.Tensor([1.0]))

        self.transformed_weight = None
        self.quantized_weight = None

        self.c_W = None
        self.d_W = None
        self.alpha_W = None
        self.beta_W = None

        self.use_cuda = torch.cuda.is_available()

        print('QIL CNN Quantization Initialization with %d-bit.' %self.bitW)

    def forward(self, x, quantize_type='STE'):

        # data_type = type(self.weight.data)
        # data_type = type(self.pruning_point.data)
        data_type = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        self.c_W = 0.5 * (self.pruning_point + self.clipping_point) # 0.5
        self.d_W = 0.5 * (self.clipping_point - self.pruning_point) # 0.5
        self.alpha_W = 0.5 / self.d_W # 1
        self.beta_W = -0.5 * self.c_W / self.d_W + 0.5 # 0

        # Weights fallen into [pruning_point, clipping_point]
        interval_weight = self.weight * \
                          (torch.abs(self.weight) > self.pruning_point).type(data_type) * \
                          (torch.abs(self.weight) < self.clipping_point).type(data_type)

        self.transformed_weight = \
            torch.sign(self.weight) * (torch.abs(self.weight) > self.clipping_point).type(data_type) + \
            torch.pow(self.alpha_W * torch.abs(interval_weight) +self.beta_W, self.gamma) * torch.sign(interval_weight)

        if quantize_type == 'interval':
            self.quantized_weight = Function_interval_quantize.apply(
                self.transformed_weight, self.pruning_point.data,
                self.clipping_point.data if self.clipping_point < 1.0 else 1.0,
                self.bitW
            )
        else:
            self.quantized_weight = 2 * Function_STE.apply(0.5 * self.transformed_weight + 0.5, self.bitW) - 1

        return F.conv2d(x, self.quantized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class QIL_QuantAct(nn.Module):

    def __init__(self, bitA=1):
        super(QIL_QuantAct, self).__init__()

        self.bitA = bitA

        self.pruning_point = nn.Parameter(torch.zeros([]))  # c_w - d_W
        self.clipping_point = nn.Parameter(torch.Tensor([1.0]))  # c_w + d_W
        # c_W = 1/2 * (pruning_point + clipping_point), d_w = 1/2 * (clipping_point - pruning_point)
        # alpha_W = 0.5 / d_w, beta_w = -0.5 * c_W / d_W + 0.5

        self.transformed_act = None
        self.quantized_act = None

        self.use_cuda = torch.cuda.is_available()

        print('QIL Activation Quantization Initialization with %d-bit.' %self.bitA)

    def forward(self, x, quantize_type='STE'):

        if self.bitA == 32:
            return x

        data_type = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_W = 0.5 * (self.pruning_point + self.clipping_point) # 0.5
        d_W = 0.5 * (self.clipping_point - self.pruning_point) # 0.5
        alpha_W = 0.5 / d_W # 1
        beta_w = -0.5 * c_W / d_W + 0.5 # 0

        interval_act = x * \
                          (torch.abs(x) > self.pruning_point).type(data_type) * \
                          (torch.abs(x) < self.clipping_point).type(data_type)

        self.transformed_act = \
            (torch.abs(x) > self.clipping_point).type(data_type) + \
            alpha_W * torch.abs(interval_act) + beta_w

        if quantize_type == 'interval':
            self.quantized_act = Function_interval_quantize.apply(
                self.transformed_act, self.pruning_point.data,
                self.clipping_point.data if self.clipping_point < 1.0 else 1.0,
                self.bitA
            )
        else:
            self.quantized_act = Function_STE.apply(self.transformed_act, self.bitA)

        return self.quantized_act


class testNet(nn.Module):

    def __init__(self):
        super(testNet, self).__init__()

        self.conv1 = QIL_CNN(3, 32, 3)

    def forward(self, x):

        return self.conv1(x)



if __name__ == '__main__':

    # net = testNet()
    # inputs = torch.rand([10, 3, 32, 32])
    # outputs = net(inputs)
    # losses = torch.mean(outputs)
    # losses.backward()

    # p = 0.3
    # c = 1
    # a = torch.clamp(torch.rand([3, 3]), min=p, max=c)
    # b = Function_interval_quantize.apply(a, p, c, 1)

    a = torch.rand([3,3])
    a[0,0] = 1.0
    a[0,1] = 0.0
    b = Function_STE.apply(a, 2)

