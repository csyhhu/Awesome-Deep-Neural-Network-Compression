import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.quantize import Function_unbiased_quantize, Function_biased_quantize, direct_biased_quantize

class dorefa_gradient_attachment(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _x, _bit, _module):
        ctx.save_for_backward(_x, _bit)
        ctx.module = _module
        return _x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _x, _bit = ctx.saved_tensors

        _max_grad = torch.max(torch.abs(grad_output))
        # [a, b] => [-1, 1] =>[0, 1]
        _shifted_grad_output = 0.5 * grad_output / _max_grad + 0.5 + 0.5 * torch.rand(size=grad_output.shape, device=_x.device) / (2 ** 8 - 1) # [alpha, \beta] => [0, 1]

        if _bit < 0 or _bit == 32:
            _normalized_quantized_grad_output, _quantized_bit_grad_output = direct_biased_quantize(_shifted_grad_output, 8)
        else:
            _normalized_quantized_grad_output, _quantized_bit_grad_output = direct_biased_quantize(_shifted_grad_output, _bit)

        grad_input = 2 * _max_grad * (_normalized_quantized_grad_output - 0.5)

        out_of_max_mask = (grad_output > _max_grad).type(type(_x)).to(_x.device)
        out_of_max_error = torch.mean(out_of_max_mask)  # Number of elements out of right threshold
        out_of_min_mask = (grad_output < -_max_grad).type(type(_x)).to(_x.device)
        out_of_min_error = torch.mean(out_of_min_mask)  # Number of elements out of left threshold
        in_domain_mask = torch.ones(grad_output.shape, device=grad_output.device) - out_of_max_mask - out_of_min_mask

        quantization_error_distribution = torch.abs(grad_input - grad_output).to(_x.device)
        quantization_error_pre_distribution = quantization_error_distribution / (torch.abs(grad_output) + 1e-8)
        # in_domain_quantization_error_pre_distribution = in_domain_mask * quantization_error_pre_distribution
        quantization_error = torch.mean(quantization_error_distribution)
        quantization_error_pre = torch.mean(quantization_error_pre_distribution)
        # in_domain_quantization_error_pre = torch.mean(in_domain_quantization_error_pre_distribution)

        in_domain_error_distribution = in_domain_mask * quantization_error_distribution  # abs(_quantized_x - _x) \leq \Delta
        in_domain_error_pre_distribution = in_domain_mask * quantization_error_pre_distribution
        in_domain_error = torch.sum(in_domain_error_distribution) / torch.sum(in_domain_mask)
        in_domain_error_pre = torch.sum(in_domain_error_pre_distribution) / torch.sum(in_domain_mask)

        _grad_quant_info = ctx.module.grad_quant_info
        for _summary_key, _summary_value in [
            ['in_domain_error', in_domain_error], ['in_domain_error_pre', in_domain_error_pre],
            ['quantization_error', quantization_error], ['quantization_error_pre', quantization_error_pre],
            # ['in_domain_quantization_error_pre', in_domain_quantization_error_pre],
            ['out_of_min_error', out_of_min_error], ['out_of_max_error', out_of_max_error],
            ['pre_quantized_grad', grad_output], ['quantized_grad', grad_input]
        ]:
            _grad_quant_info[_summary_key] = _summary_value.cpu().numpy()

        ctx.module.gradient_left_threshold = - _max_grad
        ctx.module.gradient_right_threshold = _max_grad

        if _bit == 32:
            return grad_output, None, None
        else:
            return grad_input, None, None


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

        self.quantized_weight_error = None
        self.quantized_weight_error_pre = None
        self.quantized_input_error = None
        self.quantized_input_error_pre = None

        self.max_tanh_weight = None
        self.max_input = None
        self.min_input = None

        self.use_log_scale = use_log_scale

        self.gradient_left_threshold = None
        self.gradient_right_threshold = None
        self.grad_quant_info = dict()

        print('Initialize DoReFa module with bitW=%d, bitA=%d, bitG=%d' % (bitW, bitA, bitG))

    def forward(self, x):

        # Parameter Quantization
        if self.bitW == 32:
            self.pre_quantized_weight = self.weight * 1.
            self.quantized_weight = self.pre_quantized_weight * 1.
        else:
            # [-1, 1]
            tanh_weight = torch.tanh(self.weight)
            self.max_tanh_weight = torch.max(torch.abs(tanh_weight))
            # [-1, 1]
            self.pre_quantized_weight = (tanh_weight / self.max_tanh_weight).detach()
            self.quantized_weight, self.quantized_weight_bit = Function_unbiased_quantize.apply(self.pre_quantized_weight, self.bitW)

            _quantized_weight_error_dist = torch.abs(self.quantized_weight - self.pre_quantized_weight)
            _quantized_weight_error_pre_dist = _quantized_weight_error_dist / (torch.abs(self.pre_quantized_weight) + 1e-8)
            self.quantized_weight_error = torch.mean(_quantized_weight_error_dist)
            self.quantized_weight_error_pre = torch.mean(_quantized_weight_error_pre_dist)

        # Activation Quantization
        if self.bitA == 32:
            self.pre_quantized_input = x * 1.
            self.quantized_input = self.pre_quantized_input * 1.
        else:
            # [0, 1]
            """
            tanh_input = torch.tanh(x)
            # [0, 1] or [-1, 1]
            unilateral_normalized_input = tanh_input / torch.max(torch.abs(tanh_input)).detach()
            self.min_input, self.max_input = unilateral_normalized_input.min(), unilateral_normalized_input.max()
            """
            self.min_input, self.max_input = torch.min(x).detach(), torch.max(x).detach()
            unilateral_normalized_input = x
            # Unify to [-1, 1]
            self.pre_quantized_input = (unilateral_normalized_input - self.min_input) / (self.max_input - self.min_input)
            _quantized_input, self.quantized_input_bit = Function_biased_quantize.apply(
                self.pre_quantized_input, self.bitA
            )
            # Scale to [0, 1] or [-1, 1]
            self.quantized_input = _quantized_input * (self.max_input - self.min_input) + self.min_input

            _quantized_input_error_hist = torch.abs(self.quantized_input - unilateral_normalized_input)
            _quantized_input_error_pre_hist = _quantized_input_error_hist / (torch.abs(unilateral_normalized_input) + 1e-8)
            self.quantized_input_error_pre = torch.mean(_quantized_input_error_pre_hist)
            self.quantized_input_error = torch.mean(_quantized_input_error_hist)


        output = F.conv2d(
            self.quantized_input, self.quantized_weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )

        output = dorefa_gradient_attachment.apply(output, self.bitG, self)

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
