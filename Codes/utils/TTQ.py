"""
Codes for implementing TTQ ternary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train import progress_bar, accuracy, AverageMeter

import time
import numpy as np


class Function_ternary(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, pos, neg, thresh_factor):

        thresh = thresh_factor * torch.max(torch.abs(weight))

        pos_indices = (weight > thresh).type(torch.cuda.FloatTensor)
        neg_indices = (weight < -thresh).type(torch.cuda.FloatTensor)

        ternary_weight = pos * pos_indices + neg * neg_indices

        ctx.save_for_backward(pos_indices, neg_indices, pos, neg)

        return ternary_weight

    @staticmethod
    def backward(ctx, grad_ternary_weight):

        pos_indices, neg_indices, pos, neg = ctx.saved_tensors
        pruned_indices = torch.ones(pos_indices.shape).cuda() - pos_indices - neg_indices

        grad_pos = torch.mean(grad_ternary_weight * pos_indices)
        grad_neg = torch.mean(grad_ternary_weight * neg_indices)

        grad_fp_weight = pos * grad_ternary_weight * pos_indices + \
                         grad_ternary_weight * pruned_indices + \
                         neg * grad_ternary_weight * neg_indices

        # print(grad_fp_weight.shape)

        return grad_fp_weight, grad_pos, grad_neg, None



class TTQ_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, thresh_factor=0.05):
        super(TTQ_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.pos = nn.Parameter(torch.rand([]))
        self.neg = nn.Parameter(-torch.rand([]))
        self.thresh_factor = thresh_factor

        self.ternary_weight = None
        
        
    def forward(self, x):

        self.ternary_weight = Function_ternary.apply(self.weight, self.pos, self.neg, self.thresh_factor)

        return F.conv2d(x, self.ternary_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class TTQ_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, thresh_factor=0.05):
        super(TTQ_Linear, self).__init__(in_features, out_features, bias=bias)

        self.pos = nn.Parameter(torch.rand([]))
        self.neg = nn.Parameter(-torch.rand([]))
        self.thresh_factor = thresh_factor

        self.ternary_weight = None

    def forward(self, x):

        self.ternary_weight = Function_ternary.apply(self.weight, self.pos, self.neg, self.thresh_factor)

        return F.linear(x, self.ternary_weight, self.bias)


class testNet(nn.Module):

    def __init__(self):
        super(testNet, self).__init__()
        self.conv1 = TTQ_CNN(3, 32, 5, 1, 2)

    def forward(self, x):

        return self.conv1(x)


def measure_net_stats(layer):

    ternary_weight = layer.ternary_weight.data
    pos = layer.pos.data
    neg = layer.neg.data
    n_pos = torch.sum(ternary_weight > 0).type(torch.FloatTensor)
    n_neg = torch.sum(ternary_weight < 0).type(torch.FloatTensor)
    n_prune = torch.sum(ternary_weight == 0).type(torch.FloatTensor)
    n_weight = ternary_weight.numel()

    return pos, neg, n_pos / n_weight, n_neg / n_weight, n_prune / n_weight


def test(net, test_loader, use_cuda = True, dataset_name='CIFAR10', n_batches_used=None):

    net.eval()

    if dataset_name in ['CIFAR10', 'CIFAR100', 'STL10']:
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

    elif dataset_name == 'ImageNet':

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

    else:
        raise NotImplementedError


if __name__ == '__main__':

    net = testNet()
    inputs = torch.rand([10, 3, 32, 32]).cuda()
    targets = torch.rand([10, 32, 32, 32]).cuda()

    net.cuda()
    outputs = net(inputs)
    losses = nn.MSELoss()(outputs, targets)
    losses.backward()
    print(outputs.shape)