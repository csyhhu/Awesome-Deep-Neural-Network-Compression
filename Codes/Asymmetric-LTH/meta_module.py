"""
An implementation of MetaQuant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from module import symmetric_LTH_threshold_attachment, asymmetric_LTH_threshold_attachment


class Meta_LearnableThresHold_Conv2d(nn.Conv2d):

    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1, bias=True,
        bitW=8, bitA=8,
        in_domain_grad_factor=1., out_domain_grad_factor=1.
    ):
        super(Meta_LearnableThresHold_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )

        self.bitW = bitW
        self.bitA = bitA

        self.weight_threshold = torch.nn.Parameter(torch.ones([]), requires_grad=True)
        self.left_input_threshold = torch.nn.Parameter(torch.tensor([-1.]), requires_grad=True)
        self.right_input_threshold = torch.nn.Parameter(torch.tensor([1.]), requires_grad=True)
        self.grad_scale_factor = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)

        self.quantized_weight_bit = None
        self.quantized_weight = None
        self.quantized_input_bit = None
        self.quantized_input = None

        self.max_weight = 0.
        self.min_weight = 0.
        self.max_input = 0.
        self.min_input = 0.

        self.in_domain_grad_factor = torch.nn.Parameter(torch.tensor([in_domain_grad_factor]), requires_grad=False)
        self.out_domain_grad_factor = torch.nn.Parameter(torch.tensor([out_domain_grad_factor]), requires_grad=False)

        self.fp_input = None

        print('Initialize Meta Asymmetric-LTH using bitW=%d, bitA=%d' % (self.bitW, self.bitA))

    def forward(self, x, meta_grad, lr: float):



        return F.conv2d(
            self.quantized_input, self.quantized_weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )

