'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math

from module import dorefa_Conv2d


def conv3x3(in_planes, out_planes, stride=1, bitW=8, bitA=8, bitG=8):
    " 3x3 convolution with padding "
    return dorefa_Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=False, bitW=bitW, bitA=bitA, bitG=bitG)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, layer_idx=0, block_idx=0, quantized_layer_collections=None, bitW=8, bitA=8, bitG=8):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, bitW=bitW, bitA=bitA, bitG=bitG)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bitW=bitW, bitA=bitA, bitG=bitG)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.quantized_layer_collections = quantized_layer_collections
        self.quantized_layer_collections['layer%d.%d.conv1' % (layer_idx, block_idx)] = self.conv1
        self.quantized_layer_collections['layer%d.%d.conv2' % (layer_idx, block_idx)] = self.conv2

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bitW=8, bitA=8):
        super(Bottleneck, self).__init__()
        self.conv1 = dorefa_Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False, bitW=bitW, bitA=bitA)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = dorefa_Conv2d(planes, planes, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=False, bitW=bitW, bitA=bitA)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = dorefa_Conv2d(planes, planes*4, kernel_size=(1, 1), bias=False, bitW=bitW, bitA=bitA)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10, bitW=8, bitA=8, bitG=8):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.bitW = bitW
        self.bitA = bitA
        self.bitG = bitG
        # self.conv1 = dorefa_Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False, bitW=bitW, bitA=bitA, bitG=bitG)
        # self.quantized_layer_collections = {'conv1': self.conv1}
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.quantized_layer_collections = {}
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], layer_idx=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, layer_idx=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, layer_idx=3)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, layer_idx=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                dorefa_Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=(1, 1), stride=(stride, stride), bias=False,
                    bitW=self.bitW, bitA=self.bitA, bitG=self.bitG
                ),
                nn.BatchNorm2d(planes * block.expansion)
            )
            self.quantized_layer_collections['layer%d.downsample' % layer_idx] = downsample[0]

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample,
                bitW=self.bitW, bitA=self.bitA, bitG=self.bitG,
                layer_idx=layer_idx, block_idx=0,
                quantized_layer_collections=self.quantized_layer_collections
            )
        )
        self.inplanes = planes * block.expansion
        for blk_idx in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes,
                    bitW=self.bitW, bitA=self.bitA, bitG=self.bitG,
                    layer_idx=layer_idx, block_idx=blk_idx,
                    quantized_layer_collections=self.quantized_layer_collections
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':

    import torch

    net = resnet20_cifar(bitW=4, bitA=4, bitG=4)
    # """
    inputs = torch.rand([10, 3, 32, 32])
    targets = torch.rand([10, 10])
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
    for idx in range(5):
        optimizer.zero_grad()
        outputs = net(inputs)
        losses = torch.nn.MSELoss()(outputs, targets)
        losses.backward()
        optimizer.step()
        print('[%2d] Loss: %.3e' % (idx, losses.item()))
    # """