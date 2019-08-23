"""
Helper functions and module for QIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

# from utils.quantize import Function_STE
from utils.train import progress_bar, accuracy, AverageMeter


class Function_interval_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, p, c, bitW):
        # ctx.save_for_backward(weight)
        n = float(2 ** bitW - 1)
        interval = (c-p) / n
        return torch.round((weight - p) / interval) * interval + p

    @staticmethod
    def backward(ctx, grad_outputs):

        return grad_outputs, None


class Function_STE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, bitW):
        ctx.save_for_backward(weight)

        n = float(2 ** (bitW-1) - 1)
        return torch.sign(weight) * torch.round(torch.abs(weight) * n) / n

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, = ctx.saved_tensors
        gate = (torch.abs(weight) <= 1).float()
        grad_inputs = grad_outputs * gate
        return grad_inputs, None


class QIL_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, bitW=1):
        super(QIL_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.bitW = bitW

        self.pruning_point = nn.Parameter(torch.zeros([])) # c_w - d_W
        self.clipping_point = nn.Parameter(torch.Tensor([1.0])) # c_w + d_W
        # c_W = 1/2 * (pruning_point + clipping_point), d_w = 1/2 * (clipping_point - pruning_point)
        # alpha_W = 0.5 / d_w, beta_w = -0.5 * c_W / d_W + 0.5
        self.gamma = nn.Parameter(torch.Tensor([1.0]))

        self.transformed_weight = None
        self.quantized_weight = None

        self.use_cuda = torch.cuda.is_available()

        print('QIL CNN Quantization Initialization with %d-bit.' %self.bitW)

    def forward(self, x, quantize_type='STE'):

        # data_type = type(self.weight.data)
        # data_type = type(self.pruning_point.data)
        data_type = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_W = 0.5 * (self.pruning_point + self.clipping_point) # 0.5
        d_W = 0.5 * (self.clipping_point - self.pruning_point) # 0.5
        alpha_W = 0.5 / d_W # 1
        beta_w = -0.5 * c_W / d_W + 0.5 # 0

        # Weights fallen into [pruning_point, clipping_point]
        interval_weight = self.weight * \
                          (torch.abs(self.weight) > self.pruning_point).type(data_type) * \
                          (torch.abs(self.weight) < self.clipping_point).type(data_type)

        self.transformed_weight = \
            torch.sign(self.weight) * (torch.abs(self.weight) > self.clipping_point).type(data_type) + \
            torch.pow(alpha_W * torch.abs(interval_weight) + beta_w, self.gamma) * torch.sign(interval_weight)

        if quantize_type == 'interval':
            self.quantized_weight = Function_interval_quantize.apply(
                self.transformed_weight, self.pruning_point.data,
                self.clipping_point.data if self.clipping_point < 1.0 else 1.0,
                self.bitW
            )
        else:
            self.quantized_weight = Function_STE.apply(self.transformed_weight, self.bitW)

        return F.conv2d(x, self.quantized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class QIL_QuantAct(nn.Module):

    def __init__(self, bitA=1):
        super(QIL_QuantAct, self).__init__()

        self.bitA = bitA

        self.pruning_point = nn.Parameter(torch.zeros([]))  # c_w - d_W
        self.clipping_point = nn.Parameter(torch.Tensor([1.0]))  # c_w + d_W
        # c_W = 1/2 * (pruning_point + clipping_point), d_w = 1/2 * (clipping_point - pruning_point)
        # alpha_W = 0.5 / d_w, beta_w = -0.5 * c_W / d_W + 0.5

        self.transformed_act = None
        self.quantized_act = None

        self.use_cuda = torch.cuda.is_available()

        print('QIL Activation Quantization Initialization with %d-bit.' %self.bitA)

    def forward(self, x, quantize_type='STE'):

        if self.bitA == 32:
            return x

        data_type = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_W = 0.5 * (self.pruning_point + self.clipping_point) # 0.5
        d_W = 0.5 * (self.clipping_point - self.pruning_point) # 0.5
        alpha_W = 0.5 / d_W # 1
        beta_w = -0.5 * c_W / d_W + 0.5 # 0

        interval_act = x * \
                          (torch.abs(x) > self.pruning_point).type(data_type) * \
                          (torch.abs(x) < self.clipping_point).type(data_type)

        self.transformed_act = \
            (torch.abs(x) > self.clipping_point).type(data_type) + \
            alpha_W * torch.abs(interval_act) + beta_w

        if quantize_type == 'interval':
            self.quantized_act = Function_interval_quantize.apply(
                self.transformed_act, self.pruning_point.data,
                self.clipping_point.data if self.clipping_point < 1.0 else 1.0,
                self.bitA
            )
        else:
            self.quantized_act = Function_STE.apply(self.transformed_act, self.bitA)

        return self.quantized_act


class testNet(nn.Module):

    def __init__(self):
        super(testNet, self).__init__()

        self.conv1 = QIL_CNN(3, 32, 3)

    def forward(self, x):

        return self.conv1(x)


def test(net, test_loader, use_cuda = True, dataset_name='CIFAR10', n_batches_used=None):

    net.eval()

    if dataset_name not in ['ImageNet']:
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                outputs = net(inputs)

            _, predicted = torch.max(outputs.data, dim=1)
            correct += predicted.eq(targets.data).cpu().sum().item()
            total += targets.size(0)
            progress_bar(batch_idx, len(test_loader), "Test Acc: %.3f%%" % (100.0 * correct / total))

        return 100.0 * correct / total

    else:

        batch_time = AverageMeter()
        train_loss = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            end = time.time()
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                losses = nn.CrossEntropyLoss()(outputs, targets)

                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                train_loss.update(losses.data.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % 200 == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        batch_idx, len(test_loader), batch_time=batch_time, loss=train_loss,
                        top1=top1, top5=top5))

                if n_batches_used is not None and batch_idx >= n_batches_used:
                    break

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg, top5.avg


if __name__ == '__main__':

    # net = testNet()
    # inputs = torch.rand([10, 3, 32, 32])
    # outputs = net(inputs)
    # losses = torch.mean(outputs)
    # losses.backward()

    # p = 0.3
    # c = 1
    # a = torch.clamp(torch.rand([3, 3]), min=p, max=c)
    # b = Function_interval_quantize.apply(a, p, c, 1)

    a = torch.rand([3,3])
    a[0,0] = 1.0
    a[0,1] = 0.0
    b = Function_STE.apply(a, 2)

