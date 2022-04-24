import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

from utils.quantize import Function_unbiased_quantize, Function_unbiased_quantize_STE, direct_biased_quantize

class symmetric_LTH_threshold_attachment(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _threshold, _x, _normalized_quantized_x, _grad_scale_factor, _in_domain_grad_factor, _out_domain_grad_factor):
        ctx.save_for_backward(_threshold, _x, _normalized_quantized_x, _grad_scale_factor, _in_domain_grad_factor, _out_domain_grad_factor)
        return _threshold

    @staticmethod
    def backward(ctx, grad_outputs):

        _threshold, _x, _normalized_quantized_x, \
        _grad_scale_factor, _in_domain_grad_factor, _out_domain_grad_factor, = ctx.saved_tensors

        out_of_mask = ((torch.abs(_x) - _threshold) > 0).type(type(_x)).to(_x.device) # The number of elements exceed out of threshold
        in_domain_mask = ((torch.abs(_x) - _threshold) < 0).type(type(_x)).to(_x.device)

        _quantized_x = _normalized_quantized_x * _threshold
        in_domain_error_distribution = torch.abs(_x - _quantized_x) * in_domain_mask

        # print('In domain error: %.3e | Out domain error: %.3e' % (torch.mean(in_domain_error_distribution), torch.mean(out_of_mask.type(torch.float))))

        grad_loss_wrt_threshold = _grad_scale_factor * torch.mean(
            (_in_domain_grad_factor * in_domain_error_distribution - _out_domain_grad_factor * out_of_mask)
        )
        return grad_loss_wrt_threshold, None, None, None, None, None, None

class asymmetric_LTH_threshold_attachment(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _left_threshold, _right_threshold, _x, _normalized_quantized_x, _grad_scaling_factor, _in_domain_grad_factor, _out_domain_grad_factor):
        ctx.save_for_backward(_left_threshold, _right_threshold, _x, _normalized_quantized_x, _grad_scaling_factor, _in_domain_grad_factor, _out_domain_grad_factor)
        return _right_threshold - (_left_threshold + _right_threshold) / 2.

    @staticmethod
    def backward(ctx, grad_outputs):
        _left_threshold, _right_threshold, _x, _normalized_quantized_x, \
        _grad_scaling_factor, _in_domain_grad_factor, _out_domain_grad_factor, = ctx.saved_tensors

        _mid_point = (_right_threshold + _left_threshold) / 2.
        _symmetric_threshold = _right_threshold - _mid_point
        _quantized_x = _normalized_quantized_x * _symmetric_threshold + _mid_point

        out_of_max_mask = (_x > _right_threshold).type(type(_x)).to(_x.device)
        out_of_max_error = torch.mean(out_of_max_mask)  # Number of elements out of right threshold
        out_of_min_mask = (_x < _left_threshold).type(type(_x)).to(_x.device)
        out_of_min_error = torch.mean(out_of_min_mask)  # Number of elements out of left threshold
        # print(type(out_of_max_mask))
        in_domain_mask = torch.ones(_x.shape, device=_x.device) - out_of_max_mask - out_of_min_mask

        in_domain_error_distribution = in_domain_mask * torch.abs(_quantized_x - _x).to(_x.device)  # abs(_quantized_x - _x) \leq \Delta
        in_domain_error = torch.mean(in_domain_error_distribution)

        # print(in_domain_error.device, _grad_scaling_factor.device)

        grad_loss_wrt_left_threshold = (-_in_domain_grad_factor * in_domain_error + _out_domain_grad_factor * out_of_min_error) * _grad_scaling_factor
        grad_loss_wrt_right_threshold = (_in_domain_grad_factor * in_domain_error - _out_domain_grad_factor * out_of_max_error) * _grad_scaling_factor

        return grad_loss_wrt_left_threshold, grad_loss_wrt_right_threshold, None, None, None, None, None, None, None


class asymmetric_LTH_threshold_gradient_attachment(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _left_threshold, _right_threshold, _x, _bit, _grad_scaling_factor, _in_domain_grad_factor, _out_domain_grad_factor, _module):
        ctx.save_for_backward(_left_threshold, _right_threshold, _x, _bit, _grad_scaling_factor, _in_domain_grad_factor, _out_domain_grad_factor)
        ctx.module = _module
        return _x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _left_threshold, _right_threshold, _x, _bit, \
        _grad_scaling_factor, _in_domain_grad_factor, _out_domain_grad_factor, = ctx.saved_tensors

        _clipped_grad_output = torch.clip(grad_output, _left_threshold, _right_threshold)
        _shifted_grad_output = (_clipped_grad_output - _left_threshold) / (_right_threshold - _left_threshold)
        if _bit < 0 or _bit == 32:
            _normalized_quantized_grad_output, _quantized_bit_grad_output = direct_biased_quantize(_shifted_grad_output, 8)
        else:
            _normalized_quantized_grad_output, _quantized_bit_grad_output = direct_biased_quantize(_shifted_grad_output, _bit)
        _quantized_grad_output = _normalized_quantized_grad_output * (_right_threshold - _left_threshold)  + _left_threshold

        # Record key information in gradient calculation
        out_of_max_mask = (grad_output > _right_threshold).type(type(_x)).to(_x.device)
        out_of_max_error = torch.mean(out_of_max_mask)  # Number of elements out of right threshold
        out_of_min_mask = (grad_output < _left_threshold).type(type(_x)).to(_x.device)
        out_of_min_error = torch.mean(out_of_min_mask)  # Number of elements out of left threshold
        in_domain_mask = torch.ones(grad_output.shape, device=grad_output.device) - out_of_max_mask - out_of_min_mask

        quantization_error_distribution = torch.abs(_quantized_grad_output - grad_output).to(_x.device)
        quantization_error_pre_distribution = quantization_error_distribution / (torch.abs(grad_output) + 1e-8)
        quantization_error = torch.mean(quantization_error_distribution)
        quantization_error_pre = torch.mean(quantization_error_pre_distribution)

        in_domain_error_distribution = in_domain_mask * quantization_error_distribution  # abs(_quantized_x - _x) \leq \Delta
        in_domain_error_pre_distribution = in_domain_mask * quantization_error_pre_distribution
        in_domain_error = torch.mean(in_domain_error_distribution)
        in_domain_error_pre = torch.mean(in_domain_error_pre_distribution)

        grad_loss_wrt_left_threshold = (-_in_domain_grad_factor * in_domain_error + _out_domain_grad_factor * out_of_min_error) * _grad_scaling_factor
        grad_loss_wrt_right_threshold = (_in_domain_grad_factor * in_domain_error - _out_domain_grad_factor * out_of_max_error) * _grad_scaling_factor

        _grad_quant_info = ctx.module.grad_quant_info
        for _summary_key, _summary_value in [
            ['in_domain_error', in_domain_error], ['in_domain_error_pre', in_domain_error_pre],
            ['quantization_error', quantization_error], ['quantization_error_pre', quantization_error_pre],
            ['out_of_min_error', out_of_min_error], ['out_of_max_error', out_of_max_error],
            ['pre_quantized_grad', grad_output], ['clipped_grad', _clipped_grad_output], ['quantized_grad', _quantized_grad_output],
            ['grad_loss_wrt_left_threshold', grad_loss_wrt_left_threshold], ['grad_loss_wrt_right_threshold', grad_loss_wrt_right_threshold]
        ]:
            _grad_quant_info[_summary_key] = _summary_value.cpu().numpy()

        if _bit == 32:
            return grad_loss_wrt_left_threshold, grad_loss_wrt_right_threshold, grad_output, None, None, None, None, None
        elif _bit == -1:
            return grad_loss_wrt_left_threshold, grad_loss_wrt_right_threshold, _clipped_grad_output, None, None, None, None, None
        else:
            return grad_loss_wrt_left_threshold, grad_loss_wrt_right_threshold, _quantized_grad_output, None, None, None, None, None


class asymmetric_maq_gradient_attachment(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _x, _bit, _module):
        ctx.save_for_backward(_x, _bit)
        ctx.module = _module
        return _x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _x, _bit, = ctx.saved_tensors

        _alpha = ctx.module.alpha
        _hist_max = ctx.module.hist_max
        _hist_min = ctx.module.hist_min
        _max_grad = _alpha * torch.max(torch.abs(_x)) + (1 - _alpha) * _hist_max
        _min_grad = _alpha * torch.min(torch.abs(_x)) + (1 - _alpha) * _hist_min
        ctx.module.hist_max = _max_grad
        ctx.module.hist_min = _min_grad

        _clip_grad_output = torch.clip(grad_output, min=_min_grad, max=_max_grad)
        _shifted_grad_output = (_clip_grad_output - _min_grad) / (_max_grad - _min_grad)

        if _bit < 0 or _bit == 32:
            _normalized_quantized_grad_output, _quantized_bit_grad_output = direct_biased_quantize(_shifted_grad_output, 8)
        else:
            _normalized_quantized_grad_output, _quantized_bit_grad_output = direct_biased_quantize(_shifted_grad_output, _bit)
        _quantized_grad_output = _normalized_quantized_grad_output * (_max_grad - _min_grad)  + _min_grad

        quantization_error_distribution = torch.abs(_quantized_grad_output - grad_output).to(_x.device)
        quantization_error_pre_distribution = quantization_error_distribution / (torch.abs(grad_output) + 1e-8)
        quantization_error = torch.mean(quantization_error_distribution)
        quantization_error_pre = torch.mean(quantization_error_pre_distribution)

        out_of_max_mask = (grad_output > _min_grad).type(type(_x)).to(_x.device)
        out_of_max_error = torch.mean(out_of_max_mask)  # Number of elements out of right threshold
        out_of_min_mask = (grad_output < _max_grad).type(type(_x)).to(_x.device)
        out_of_min_error = torch.mean(out_of_min_mask)  # Number of elements out of left threshold
        in_domain_mask = torch.ones(grad_output.shape, device=grad_output.device) - out_of_max_mask - out_of_min_mask

        in_domain_error_distribution = in_domain_mask * quantization_error_distribution  # abs(_quantized_x - _x) \leq \Delta
        in_domain_error_pre_distribution = in_domain_mask * quantization_error_pre_distribution
        in_domain_error = torch.mean(in_domain_error_distribution)
        in_domain_error_pre = torch.mean(in_domain_error_pre_distribution)

        _grad_quant_info = ctx.module.grad_quant_info
        for _summary_key, _summary_value in [
            ['pre_quantized_grad', grad_output], ['clipped_grad', _clip_grad_output], ['quantized_grad', _quantized_grad_output],
            ['quantization_error', quantization_error], ['quantization_error_pre', quantization_error_pre],
            ['in_domain_error', in_domain_error], ['in_domain_error_pre', in_domain_error_pre],
            ['out_of_min_error', out_of_min_error], ['out_of_max_error', out_of_max_error],
        ]:
            _grad_quant_info[_summary_key] = _summary_value.cpu().numpy()

        return _quantized_grad_output, None, None


class LearnableThresHold_Conv2d(nn.Conv2d):

    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1, bias=True,
        bitW=8, bitA=8, bitG=8,
        gradient_quantized_type='alth',
        in_domain_grad_factor=1.,
        out_of_domain_kernel=1., out_of_domain_activation=1., out_of_domain_gradient=1.
    ):
        super(LearnableThresHold_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )

        self.bitW = bitW
        self.bitA = bitA
        self.bitG = torch.tensor(bitG)
        self.gradient_quantized_type = gradient_quantized_type

        self.weight_threshold = torch.nn.Parameter(torch.ones([]), requires_grad=True)
        self.left_input_threshold = torch.nn.Parameter(torch.tensor([-1.]), requires_grad=True)
        self.right_input_threshold = torch.nn.Parameter(torch.tensor([1.]), requires_grad=True)
        self.grad_scale_factor = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)

        # if self.bitG != 32:
        self.gradient_left_threshold = torch.nn.Parameter(torch.tensor([-1e-2]), requires_grad=True)
        self.gradient_right_threshold = torch.nn.Parameter(torch.tensor([1e-2]), requires_grad=True)

        self.quantized_weight_bit = None
        self.quantized_weight = None
        self.quantized_input_bit = None
        self.quantized_input = None

        self.max_weight = 0.
        self.min_weight = 0.
        self.max_input = 0.
        self.min_input = 0.

        self.in_domain_grad_factor = torch.nn.Parameter(torch.tensor([in_domain_grad_factor]), requires_grad=False)
        self.out_of_domain_kernel = torch.nn.Parameter(torch.tensor([out_of_domain_kernel]), requires_grad=False)
        self.out_of_domain_activation = torch.nn.Parameter(torch.tensor([out_of_domain_activation]), requires_grad=False)
        self.out_of_domain_gradient = torch.nn.Parameter(torch.tensor([out_of_domain_gradient]), requires_grad=False)

        self.fp_input = None

        self.grad_quant_info = dict()
        self.hist_max = 0.
        self.hist_min = 0.
        self.alpha = 0.9

        print(
            'Initialize asymmetric-LTH using bitW=%d, bitA=%d, bitG=%d (%s), out-of-domain penalty: %.2e/%.2e/%.2e' % (
                self.bitW, self.bitA, bitG, self.gradient_quantized_type,
                out_of_domain_kernel, out_of_domain_activation, out_of_domain_gradient
            )
        )

    def forward(self, x):

        self.fp_input = x

        self.max_weight = self.weight.data.max()
        self.min_weight = self.weight.data.min()
        self.max_input = x.data.max()
        self.min_input = x.data.min()

        if self.bitW == 32:
            self.quantized_weight = self.weight * 1.
        else:
            # Quantize weight
            # [-\alpha, \alpha]
            _clip_weight = torch.clip(self.weight, min=-self.weight_threshold.data, max=self.weight_threshold.data)
            # _clip_mask = self.weight_threshold.data > self.weight.data > - self.weight_threshold.data
            _clip_mask = torch.le(torch.abs(self.weight.data), self.weight_threshold.data).to(self.weight.device)
            # [-1, 1]
            _normalized_weight = _clip_weight / self.weight_threshold.data
            _normalized_quantized_weight, self.quantized_weight_bit = Function_unbiased_quantize_STE.apply(
                _normalized_weight, self.bitW, _clip_mask
            )
            self.quantized_weight = _normalized_quantized_weight * symmetric_LTH_threshold_attachment.apply(
                self.weight_threshold,
                self.weight.data,
                _normalized_quantized_weight,
                self.grad_scale_factor,
                self.in_domain_grad_factor, self.out_of_domain_kernel
            )

        if self.bitA == 32:
            self.quantized_input = x * 1.
        else:
            # Quantize input
            _clip_x = torch.clip(x, min=self.left_input_threshold.data, max=self.right_input_threshold.data)
            # _clip_mask = self.left_input_threshold.data < x.data < self.right_input_threshold.data
            _clip_mask = torch.logical_and(
                torch.le(x.data, self.right_input_threshold.data),
                torch.ge(x.data, self.left_input_threshold.data)
            ).to(x.device)
            #
            _mid_point = ((self.left_input_threshold + self.right_input_threshold) / 2.).data
            _symmetric_threshold = (self.right_input_threshold - _mid_point).data
            _shift_scale_x = (_clip_x - _mid_point) / _symmetric_threshold
            # [-1, 1] => [-2^{b-1}, 2^{b-1}] => [-1, 1]
            _normalized_quantized_input, self.quantized_input_bit = Function_unbiased_quantize_STE.apply(
                _shift_scale_x, self.bitA, _clip_mask
            )
            # [-1, 1] => [a, b]
            self.quantized_input = _normalized_quantized_input * asymmetric_LTH_threshold_attachment.apply(
                self.left_input_threshold,
                self.right_input_threshold,
                x,
                _normalized_quantized_input,
                self.grad_scale_factor,
                self.in_domain_grad_factor, self.out_of_domain_activation
            ) + _mid_point

        output = F.conv2d(
            self.quantized_input, self.quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        # if self.bitG != 32:
        if self.gradient_quantized_type == 'alth':
            output = asymmetric_LTH_threshold_gradient_attachment.apply(
                self.gradient_left_threshold, self.gradient_right_threshold,
                output, self.bitG,
                self.grad_scale_factor,
                self.in_domain_grad_factor,
                self.out_of_domain_gradient,
                self
            )
        elif self.gradient_quantized_type == 'maq':
            output = asymmetric_maq_gradient_attachment.apply(
                output, self.bitG,
                self
            )

        return output

if __name__ == '__main__':

    import torch
    import numpy as np

    # bn = torch.nn.BatchNorm2d()
    batch_size = 10
    in_channels = 32
    out_channels = 32
    kernel_size = 3
    padding = 1
    width = 28
    height = 28
    bitW = 2
    bitA = 2
    bitG = 2
    conv = LearnableThresHold_Conv2d(in_channels, out_channels, (kernel_size, kernel_size), padding ='same', bitW=bitW, bitA=bitA, bitG=bitG, gradient_quantized_type='maq')
    optimizer = torch.optim.SGD(conv.parameters(), lr=1e-3)
    inputs = (torch.rand([batch_size, in_channels, width, height]) - 0.5 ) * 2.5
    targets = torch.rand([batch_size, in_channels, width, height])
    for idx in range(5):

        print('[%2d] weight threshold: %.3e ' % (idx, conv.weight_threshold.data.numpy()))

        optimizer.zero_grad()
        outputs = conv(inputs)
        losses = torch.nn.MSELoss()(outputs, targets)
        losses.backward()
        optimizer.step()

        print(
            '[%2d] weight threshold: %.3e (%.3e) | left input threshold: %.3e (%.3e) | right input threshold: %.3e (%.3e) | gradient left/right threshold: %3e/%.3e' % (
                idx,
                conv.weight_threshold.data.numpy(), conv.weight_threshold.grad.item(),
                conv.left_input_threshold.data.numpy(), conv.left_input_threshold.grad.item(),
                conv.right_input_threshold.data.numpy(), conv.right_input_threshold.grad.item(),
                conv.gradient_left_threshold.data.numpy(), conv.gradient_right_threshold.data.numpy()
            )
        )