import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.quantize import Function_unbiased_quantize

class MAQ_XNOR_Conv2d(nn.Conv2d):

    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1, bias=True,
        bitW=8, bitA=8,
        alpha=0.9
    ):
        super(MAQ_XNOR_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )

        self.bitW = bitW
        self.bitA = bitA
        self.alpha = alpha
        self.register_buffer("weight_threshold", torch.zeros([]))
        self.register_buffer("left_input_threshold", torch.tensor([-1.]))
        self.register_buffer("right_input_threshold", torch.tensor([1.]))

        self.quantized_weight_bit = None
        self.quantized_weight = None
        self.quantized_input_bit = None
        self.quantize_input = None

        print('Initialize asymmetric-MAQ-XNOR using bitW=%d, bitA=%d, alpha=%.2f' % (self.bitW, self.bitA, self.alpha))

    def forward(self, x):

        if self.training:
            self.weight_threshold = torch.max(torch.abs(self.weight.data))
            self.left_input_threshold = self.alpha * torch.min(x) + (1 - self.alpha) * self.left_input_threshold.data
            self.right_input_threshold = self.alpha * torch.max(x) + (1 - self.alpha) * self.right_input_threshold.data

        if self.bitW == 32:
            self.quantized_weight = self.weight * 1.
        else:
            # Quantize weight
            # [-\alpha, \alpha]
            _clip_weight = torch.clip(self.weight, min=-self.weight_threshold, max=self.weight_threshold)
            # [-1, 1]
            _normalized_weight = _clip_weight / self.weight_threshold.data
            _normalized_quantized_weight, self.quantized_weight_bit = Function_unbiased_quantize.apply(_normalized_weight, self.bitW)
            self.quantized_weight = _normalized_quantized_weight * self.weight_threshold.data

        if self.bitA == 32:
            self.quantize_input = x * 1.
        else:
            # Quantize input
            _clip_x = torch.clip(x, min=self.left_input_threshold, max=self.right_input_threshold)
            #
            _mid_point = ((self.left_input_threshold + self.right_input_threshold) / 2.).data
            _symmetric_threshold = (self.right_input_threshold - _mid_point).data
            _shift_scale_x = (_clip_x - _mid_point) / _symmetric_threshold
            # [-1, 1] => [-2^{b-1}, 2^{b-1}] => [-1, 1]
            _normalized_quantized_input, self.quantized_input_bit = Function_unbiased_quantize.apply(_shift_scale_x, self.bitA)
            # [-1, 1] => [a, b]
            self.quantize_input = _normalized_quantized_input * _symmetric_threshold + _mid_point

        return F.conv2d(
            self.quantize_input, self.quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
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
    conv = MAQ_XNOR_Conv2d(in_channels, out_channels, (kernel_size, kernel_size), padding = 'same' , bitW=bitW, bitA=bitA)
    optimizer = torch.optim.SGD(conv.parameters(), lr=1e-1)
    inputs = (torch.rand([batch_size, in_channels, width, height]) - 0.5 ) * 2.5
    targets = torch.rand([batch_size, in_channels, width, height])
    for idx in range(5):
        optimizer.zero_grad()
        outputs = conv(inputs)
        losses = torch.nn.MSELoss()(outputs, targets)
        losses.backward()
        optimizer.step()
        print(
            '[%2d] weight threshold: %.3e | left input threshold: %.3e | right input threshold: %.3e' % (
                idx,
                conv.weight_threshold.data.numpy(),
                conv.left_input_threshold.data.numpy(),
                conv.right_input_threshold.data.numpy()
            )
        )
        print('Quantized weight: %s' % (np.unique(conv.quantized_weight.data.numpy())))
        print('Quantized input: %s' % (np.unique(conv.quantize_input.data.numpy())))