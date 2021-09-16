import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.quantize import Function_unbiased_quantize

class dorefa_Conv2d(nn.Conv2d):

    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1, bias=True,
        bitW=8, bitA=8
    ):
        super(dorefa_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )

        self.bitW = bitW
        self.bitA = bitA

        self.pre_quantized_weight = None
        self.quantized_weight = None
        self.quantized_weight_bit = None
        self.pre_quantized_input = None
        self.quantized_input = None
        self.quantized_input_bit = None


    def forward(self, x):

        # Parameter Quantization
        if self.bitW == 32:
            self.pre_quantized_weight = self.weight * 1.
            self.quantized_weight = self.pre_quantized_weight * 1.
        else:
            # [-1, 1]
            temp_weight = torch.tanh(self.weight)
            # [-1, 1]
            self.pre_quantized_weight = (temp_weight / torch.max(torch.abs(temp_weight)).detach())
            self.quantized_weight, self.quantized_weight_bit = Function_unbiased_quantize.apply(self.pre_quantized_weight, self.bitW)

        # Activation Quantization
        if self.bitA == 32:
            self.pre_quantized_input = x * 1.
            self.quantized_input = self.pre_quantized_input * 1.
        else:
            # [0, 1]
            temp_input = torch.tanh(x)
            # [0, 1] or [-1, 1]
            unilateral_normalized_input = temp_input / torch.max(torch.abs(temp_input)).detach()
            min_uni, max_uni = unilateral_normalized_input.min(), unilateral_normalized_input.max()
            # Unify to [-1, 1]
            self.pre_quantized_input = (unilateral_normalized_input - min_uni) / (max_uni - min_uni)
            _quantized_input, self.quantized_input_bit = Function_unbiased_quantize.apply(
                self.pre_quantized_input, self.bitA
            )
            # Scale to [0, 1] or [-1, 1]
            self.quantized_input = _quantized_input * (max_uni - min_uni) + min_uni

        return F.conv2d(
            self.quantized_input, self.quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )