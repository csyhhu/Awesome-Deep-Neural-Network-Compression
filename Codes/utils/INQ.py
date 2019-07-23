"""
Codes for implementing Incremental Network Quantization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Function_stopGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, stopGradientMask):
        ctx.save_for_backward(stopGradientMask)
        return weight

    @staticmethod
    def backward(ctx, grad_outputs):
        stopGradientMask, = ctx.saved_tensors
        grad_inputs = grad_outputs * stopGradientMask
        return grad_inputs, None


class INQ_Conv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(INQ_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.mask_weight = None

    def forward(self, x, stopGradientMask=None):

        if stopGradientMask is None:
            self.mask_weight = self.weight * 1.0
        else:
            self.mask_weight = Function_stopGradient.apply(self.weight, stopGradientMask)

        return F.conv2d(x, self.mask_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class INQ_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(INQ_Linear, self).__init__(in_features, out_features, bias=bias)

        self.mask_weight = None

    def forward(self, x, stopGradientMask=None):

        if stopGradientMask is None:
            self.mask_weight = self.weight * 1.0
        else:
            self.mask_weight = Function_stopGradient.apply(self.weight, stopGradientMask)

        return F.linear(x, self.mask_weight, self.bias)


def quantize(param, bitW=5):
    # Maximum index
    n1 = torch.floor(torch.log2(4 * torch.max(param) / 3))
    # Minmum index
    n2 = n1 + 1 - 2**(bitW - 2)
    # for n in range(int(n2.numpy()), int(n1.numpy())):
    #     print(2**n)
    sign = torch.sign(param)
    exponent = torch.clamp(torch.floor(torch.log2(torch.abs(param))), max=n1)
    # print(exponent)
    q_value = 2**exponent
    q_value[q_value < 2**n2] = 0
    q_value *= sign
    return q_value


def partition_quantize_weight(weight, mask, quant_ratio):
    """
    This function partition the remaining weights (1 in mask_grad_dict) into quantized part and fp part,
    then quantize corresponding part into 2^p
    weight: tensor
    mask: tensor
    """
    # Get remaining weights (mask as 1)
    remaining_weights = weight * mask
    n_elements = weight.numel()
    n_remaining_elements = torch.sum(mask)
    # Number of elements to be quantized in this iteration
    # No.Remaining Elements - No.Require.Remaining Elements
    n_quant = int(n_remaining_elements - n_elements * (1 - quant_ratio / 100.0))
    # Higher than quantization point will be quantized
    quant_point = torch.topk(torch.abs(remaining_weights).view(-1), n_quant)[0][-1]
    quantized_index = torch.abs(remaining_weights) >= quant_point
    # Weights that be quantized
    quantized_weights = remaining_weights * quantized_index.float()
    # Quantized weight
    q_weights = quantize(quantized_weights)
    # First set quantized weights to 0, then add quantized weights back
    quantWeight = weight.clone()
    quantWeight[quantized_index] = 0
    quantWeight += q_weights
    new_mask = mask.clone()
    new_mask[quantized_index] = 0

    return quantWeight, new_mask


if __name__ == '__main__':

    weights = torch.rand([3, 32, 7, 7])
    stopGradientMask = torch.ones(weights.shape)
    quantWeight, new_mask = partition_quantize_weight(weights, stopGradientMask, 50)