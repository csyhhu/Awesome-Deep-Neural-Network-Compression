"""
A set of helper functions for quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train import progress_bar, accuracy, AverageMeter

import time
import numpy as np

class Function_STE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, bitW):
        ctx.save_for_backward(weight)

        n = float(2 ** bitW - 1)
        return torch.round(weight * n) / n

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, = ctx.saved_tensors
        gate = (torch.abs(weight) <= 1).float()
        grad_inputs = grad_outputs * gate
        return grad_inputs, None


class Function_sign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        return torch.sign(weight)

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, = ctx.saved_tensors
        gate = (torch.abs(weight) <= 1).float()
        grad_inputs = grad_outputs * gate
        return grad_inputs, None


class Function_unbiased_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _normalized_weight, _bit):
        ctx.save_for_backward(_normalized_weight)
        n = 2 ** (_bit - 1)
        _round_x = torch.round(_normalized_weight * n)  # [-1, 1] => [-n, n]
        # {-n, n, 1} => {-n, n-1, 1}
        _quantized_bit = torch.clip(
            _round_x, -n, n - 1
        )
        return _quantized_bit / n, _quantized_bit

    @staticmethod
    def backward(ctx, _grad_normalized_quantized_weight, _grad_quantized_bit):
        return _grad_normalized_quantized_weight, None


class quantized_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, bitW = 1):

        super(quantized_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.quantized_weight = None
        self.pre_quantized_weight = None
        self.bitW = bitW
        self.alpha = None

        self.quantized_grads = None
        
        print('Initial quantized CNN with bit %d' %self.bitW)


    def forward(self, input, quantized):

        if quantized == 'dorefa':
            temp_weight = torch.tanh(self.weight)
            self.pre_quantized_weight = (temp_weight / torch.max(torch.abs(temp_weight)).detach()) * 0.5 + 0.5
            self.quantized_weight = 2 * Function_STE.apply(self.pre_quantized_weight, self.bitW) - 1
        elif quantized == 'BWN':
            self.alpha = torch.mean(torch.abs(self.weight.data))
            self.pre_quantized_weight = self.weight / self.alpha.data
            self.quantized_weight = self.alpha * Function_sign.apply(self.pre_quantized_weight)
        elif quantized == 'BWN-F':
            self.alpha = torch.abs(self.weight.data).mean(-1).mean(-1).mean(-1).view(-1, 1, 1, 1)
            self.pre_quantized_weight = self.weight / self.alpha.data
            self.quantized_weight = self.alpha * Function_sign.apply(self.pre_quantized_weight)
        else:
            self.quantized_weight = self.weight * 1.0

        return F.conv2d(input, self.quantized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class quantized_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, bitW = 1):
        super(quantized_Linear, self).__init__(in_features, out_features, bias=bias)

        self.quantized_weight = None
        self.pre_quantized_weight = None
        self.bitW = bitW
        self.alpha = None
        print('Initial quantized Linear with bit %d' % self.bitW)

    def forward(self, input, quantized):

        if quantized == 'dorefa':
            temp_weight = torch.tanh(self.weight)
            self.pre_quantized_weight = (temp_weight / torch.max(torch.abs(temp_weight)).detach()) * 0.5 + 0.5
            self.quantized_weight = 2 * Function_STE.apply(self.pre_quantized_weight, self.bitW) - 1
        elif quantized in ['BWN', 'BWN-F']:
            self.alpha = torch.mean(torch.abs(self.weight.data))
            self.pre_quantized_weight = self.weight / self.alpha.data
            self.quantized_weight = self.alpha * Function_sign.apply(self.pre_quantized_weight)
        else:
            self.quantized_weight = self.weight * 1.0

        return F.linear(input, self.quantized_weight, self.bias)


def test(net, quantized_type, test_loader, use_cuda = True, dataset_name='CIFAR10', n_batches_used=None):

    net.eval()

    if dataset_name != 'ImageNet':

        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                outputs = net(inputs, quantized_type)

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
                outputs = net(inputs, quantized_type)
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
    pass