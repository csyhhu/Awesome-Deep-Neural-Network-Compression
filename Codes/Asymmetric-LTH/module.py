import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.quantize import Function_unbiased_quantize, Function_unbiased_quantize_STE

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

        grad_loss_wrt_left_threshold = (
                                           -_in_domain_grad_factor * in_domain_error +
                                           _out_domain_grad_factor * out_of_min_error
                                       ) * _grad_scaling_factor
        grad_loss_wrt_right_threshold = (
                                            _in_domain_grad_factor * in_domain_error -
                                            _out_domain_grad_factor * out_of_max_error
                                        ) * _grad_scaling_factor

        return grad_loss_wrt_left_threshold, grad_loss_wrt_right_threshold, None, None, None, None, None, None

class LearnableThresHold_Conv2d(nn.Conv2d):

    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1, bias=True,
        bitW=8, bitA=8,
        in_domain_grad_factor=1., out_domain_grad_factor=0.7
    ):
        super(LearnableThresHold_Conv2d, self).__init__(
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

        print('Initialize asymmetric-LTH using bitW=%d, bitA=%d' % (self.bitW, self.bitA))

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
                self.in_domain_grad_factor, self.out_domain_grad_factor
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
                self.in_domain_grad_factor, self.out_domain_grad_factor
            ) + _mid_point

        return F.conv2d(
            self.quantized_input, self.quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


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
    conv = LearnableThresHold_Conv2d(in_channels, out_channels, (kernel_size, kernel_size), padding ='same', bitW=bitW, bitA=bitA)
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
            '[%2d] weight threshold: %.3e (%.3e) | left input threshold: %.3e (%.3e) | right input threshold: %.3e (%.3e)' % (
                idx,
                conv.weight_threshold.data.numpy(), conv.weight_threshold.grad.item(),
                conv.left_input_threshold.data.numpy(), conv.left_input_threshold.grad.item(),
                conv.right_input_threshold.data.numpy(), conv.right_input_threshold.grad.item()
            )
        )
        # print('Quantized weight: %s' % (np.unique(conv.quantized_weight.data.numpy())))
        # print('Quantized input: %s' % (np.unique(conv.quantize_input.data.numpy())))