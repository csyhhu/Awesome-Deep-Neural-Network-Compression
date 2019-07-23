'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math
from utils.quantize import quantized_CNN, quantized_Linear


def conv3x3(in_planes, out_planes, stride=1, bitW=1):
    " 3x3 convolution with padding "
    return quantized_CNN(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                         bitW=bitW)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bitW=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, bitW=bitW)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bitW=bitW)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, quantized=None):
        residual = x

        out = self.conv1(x, quantized)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, quantized)
        out = self.bn2(out)

        if self.downsample is not None:
            # residual = self.downsample(x)
            for module in self.downsample:
                if isinstance(module, quantized_CNN):
                    residual = module(residual, quantized)
                else:
                    residual = module(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bitW=1):
        super(Bottleneck, self).__init__()
        self.conv1 = quantized_CNN(inplanes, planes, kernel_size=1, bias=False, bitW=bitW)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = quantized_CNN(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, bitW=bitW)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = quantized_CNN(planes, planes*4, kernel_size=1, bias=False, bitW=bitW)
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

    def __init__(self, block, layers, first_stride=1, num_classes=10, bitW=1):
        super(ResNet_Cifar, self).__init__()
        self.bitW = bitW
        self.inplanes = 16
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = quantized_CNN(3, 16, kernel_size=3, stride=first_stride, padding=1, bias=False,
                              bitW=self.bitW)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = quantized_Linear(64 * block.expansion, num_classes, bitW=self.bitW)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                quantized_CNN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                              bitW=self.bitW),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, bitW=self.bitW))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, bitW=self.bitW))

        return nn.Sequential(*layers)

    def forward(self, x, quantized=None):

        x = self.conv1(x, quantized)
        x = self.bn1(x)
        x = self.relu(x)
        for layer in [self.layer1, self.layer2, self.layer3]:
            for module in layer:
                x = module(x, quantized)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, quantized)

        return x


class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet20_cifar(bitW=1, num_classes=10):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], num_classes = num_classes, bitW=bitW)
    return model


def resnet20_stl(bitW=1, num_classes=10):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], first_stride=3, num_classes = num_classes, bitW=bitW)
    return model


def resnet32_cifar(bitW=1, num_classes=10):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], num_classes = num_classes, bitW=bitW)
    return model


def resnet44_cifar(bitW=1, num_classes=10):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], num_classes = num_classes, bitW=bitW)
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
    # net = preact_resnet110_cifar()
    net = resnet20_cifar()
    y = net(torch.autograd.Variable(torch.randn(1, 3, 32, 32)))
    print(net)
    print(y.size())

