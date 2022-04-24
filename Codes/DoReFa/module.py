import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.quantize import Function_unbiased_quantize, direct_biased_quantize

class dorefa_gradient_attachment(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _x, _bit):
        ctx.save_for_backward(_x, _bit)
        return _x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _x, _bit, = ctx.saved_tensors

        _max_grad = torch.max(torch.abs(grad_output))
        _shifted_grad_output = 0.5 * grad_output / _max_grad + 0.5 # [alpha, \beta] => [0, 1]

        if _bit < 0 or _bit == 32:
            _normalized_quantized_grad_output, _quantized_bit_grad_output = direct_biased_quantize(_shifted_grad_output, 8)
        else:
            _normalized_quantized_grad_output, _quantized_bit_grad_output = direct_biased_quantize(_shifted_grad_output, _bit)

        grad_input = 2 * _max_grad * (_normalized_quantized_grad_output - 0.5)

        if _bit == 32:
            return grad_output, None
        else:
            return grad_input, None


class dorefa_Conv2d(nn.Conv2d):

    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1, bias=True,
        bitW=8, bitA=8, bitG=8, use_log_scale=False
    ):
        super(dorefa_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )

        self.bitW = bitW
        self.bitA = bitA
        self.bitG = torch.tensor(bitG)

        self.pre_quantized_weight = None
        self.quantized_weight = None
        self.quantized_weight_bit = None
        self.pre_quantized_input = None
        self.quantized_input = None
        self.quantized_input_bit = None

        self.use_log_scale = use_log_scale

        print('Initialize DoReFa module with bitW=%d, bitA=%d, bitG=%d' % (bitW, bitA, bitG))

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

        output = F.conv2d(
            self.quantized_input, self.quantized_weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )

        output = dorefa_gradient_attachment.apply(output, self.bitG)

        return output


if __name__ == '__main__':

    import torch

    batch_size = 10
    in_channels = 3
    out_channels = 3
    bitW = 4
    bitA = 4
    bitG = 4

    module = dorefa_Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=(3, 3), padding=(1, 1),
        bitA=bitA, bitW=bitW, bitG=bitG
    )

    inputs = torch.rand(size=(batch_size, in_channels, 32, 32))
    outputs = module(inputs)
    targets = torch.rand(size=[batch_size, out_channels, 32, 32])
    losses = torch.nn.MSELoss()(targets, outputs)
    losses.backward()
