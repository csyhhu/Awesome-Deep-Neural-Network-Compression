"""
Codes for implementing Incremental Network Quantization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.miscellaneous import get_layer

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


def quantize(param, n1=-1, bitW=5):
    # Maximum index
    # n1 = torch.floor(torch.log2(4 * torch.max(torch.abs(param)) / 3))
    # Minmum index
    # if bitW == 1:
    #     n2 = n1 - 1
    # else:
    #     n2 = n1 + 1 - 2**(bitW - 1) / 2
    n2 = n1 - bitW
    # print('n2: %d' %n2)
    # for n in range(int(n2.numpy()), int(n1.numpy())):
    #     print(2**n)
    sign = torch.sign(param)
    exponent = torch.clamp(torch.floor(torch.log2(torch.abs(param))), max=(n1-1)) # 2^n_1 is not accessiable
    # print(exponent)
    q_value = 2**exponent
    q_value[q_value < 2**n2] = 0
    q_value *= sign
    return q_value


def partition_quantize_weight(weight, mask, quant_ratio, n1=-1, bitW=5):
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
    q_weights = quantize(quantized_weights, n1=n1, bitW=bitW)
    # First set quantized weights to 0, then add quantized weights back
    quantWeight = weight.clone()
    quantWeight[quantized_index] = 0
    quantWeight += q_weights
    new_mask = mask.clone()
    new_mask[quantized_index] = 0

    return quantWeight, new_mask


def check_INQ_bits(net):

    for (layer_name, layer_info) in net.layer_name_list:
        weight = get_layer(net, layer_info).weight.data
        n_unique = len(torch.unique(weight))
        print('Number of unique bits in %s: %d' %(layer_name, n_unique))


class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.conv1 = INQ_Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.layer_name_list = [['conv1', 'conv1']]

        # self.conv1.weight.data.copy_(2 * (torch.rand([32, 3, 3, 3]) - 0.5))

    def forward(self, x, mask):
        return self.conv1(x, mask)


if __name__ == '__main__':

    # weights = 2*(torch.rand([10, 10]) - 0.5).cuda()
    # stopGradientMask = torch.ones(weights.shape).cuda()
    # quantWeight, new_mask = partition_quantize_weight(weights, stopGradientMask, 50)

    net = testNet()
    stopGradientMaskDict = dict()
    n1_dict = dict()
    for layer_name, layer_info in net.layer_name_list:
        layer_weight = get_layer(net, layer_info).weight.data
        stopGradientMaskDict[layer_name] = torch.ones(layer_weight.shape)
        n1_dict[layer_name] = torch.floor(torch.log2(4 * torch.max(torch.abs(layer_weight)) / 3))

    print('n1: ')
    print(n1_dict)

    print('1-th Weights before update')
    print(net.conv1.weight.data[0, 0, :, :])

    for layer_name, layer_info in net.layer_name_list:
        layer_weight = get_layer(net, layer_info).weight
        quantWeight, new_stopGradientMask = \
            partition_quantize_weight(layer_weight.data,
                                      stopGradientMaskDict[layer_name], n1=n1_dict[layer_name],
                                      quant_ratio=50, bitW=1)
        layer_weight.data.copy_(quantWeight)
        stopGradientMaskDict[layer_name] = new_stopGradientMask

    print('Mask')
    print(stopGradientMaskDict['conv1'][0, 0, :, :])

    for batch_idx in range(10):
        inputs = torch.rand([10, 3, 32, 32])
        targets = torch.rand([10, 32, 30, 30])

        optimizer = torch.optim.SGD(net.parameters(), lr=1)
        outputs = net(inputs, stopGradientMaskDict['conv1'])
        losses = torch.nn.MSELoss()(outputs, targets)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print('1-th Weights after update')
    print(net.conv1.weight.data[0, 0, :, :])
    print(len(torch.unique(net.conv1.weight.data)))

    for layer_name, layer_info in net.layer_name_list:
        layer_weight = get_layer(net, layer_info).weight
        quantWeight, new_stopGradientMask = \
            partition_quantize_weight(layer_weight.data,
                                      stopGradientMaskDict[layer_name], n1=n1_dict[layer_name],
                                      quant_ratio=100, bitW=1)
        layer_weight.data.copy_(quantWeight)
        stopGradientMaskDict[layer_name] = new_stopGradientMask

    # print('Mask')
    # print(stopGradientMaskDict['conv1'][0, 0, :, :])

    ########################################################################

    for batch_idx in range(10):
        inputs = torch.rand([10, 3, 32, 32])
        targets = torch.rand([10, 32, 30, 30])

        optimizer = torch.optim.SGD(net.parameters(), lr=1)
        outputs = net(inputs, stopGradientMaskDict['conv1'])
        losses = torch.nn.MSELoss()(outputs, targets)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print('2-th Weights after update')
    print(net.conv1.weight.data[0, 0, :, :])
    print(torch.unique(net.conv1.weight.data))