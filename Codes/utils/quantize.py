"""
A set of helper functions for quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train import progress_bar, accuracy, AverageMeter

import time
import numpy as np


def gaussian(x, mu, sigma):

    return 1 / (sigma * torch.sqrt(torch.tensor(2 * np.pi))) * torch.exp(-0.5 * torch.pow((x - mu / sigma), 2))


def direct_biased_quantize(x, _bit):
    """
    Quantize x (within [0, 1] to discrete value), without processing the gradient
    Args:
        x:
        _bit:

    Returns:

    """
    if _bit == 32:
        return x, x
    else:
        n = 2 ** _bit  - 1
        _round_x = torch.round(x * n)  # [0, 1] => [0, n]
        # {-n, n, 1} => {-n, n-1, 1}
        _quantized_bit = torch.clip(
            _round_x, 0, n
        )
        return _quantized_bit / n, _quantized_bit


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


class Function_biased_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _normalized_weight, _bit):
        """
        Args:
            ctx:
            _normalized_weight: weights within [-1, 1]
            _bit:
        Returns:
        """
        ctx.save_for_backward(_normalized_weight)
        n = 2 ** _bit - 1
        _round_x = torch.round(_normalized_weight * n)  # [-1, 1] => [-n, n]
        # {-n, n, 1} => {-n, n-1, 1}
        _quantized_bit = torch.clip(
            _round_x, 0, n - 1
        )
        return _quantized_bit / n, _quantized_bit

    @staticmethod
    def backward(ctx, _grad_normalized_quantized_weight, _grad_quantized_bit):
        return _grad_normalized_quantized_weight, None


class Function_biased_log2_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _normalized_x, _bit):
        ctx.save_for_backward(_normalized_x)
        n = 2 ** _bit
        # uni_range = torch.arange(0, n) / n
        # [0, 1] => [1, 2] => [0, 1]
        log_x = torch.log2(_normalized_x + 1)
        _round_x = torch.round(log_x * n)  # [0, 1] => [0, n] => {0, n-1}
        _quantized_bit = torch.clip(_round_x, 0, n - 1)
        # {0, n-1} => {0, 1} => {1, 2} => {0, 1}
        _quantized_x = 2 ** (_quantized_bit / n) - 1

        return _quantized_x, _quantized_bit

    @staticmethod
    def backward(ctx, _grad_normalized_quantized_weight, _grad_quantized_bit):
        return _grad_normalized_quantized_weight, None



class Function_log_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _normalized_x, _bit):
        ctx.save_for_backward(_normalized_x)
        n = 2 ** (_bit - 1)
        sign = torch.sign(_normalized_x)
        # [-1, 1] => [0, 1]
        _abs_x = torch.abs(_normalized_x)
        # [0, 1] => [1, 2]
        _shift_x = _abs_x + 1
        # [1, 2] => [0, log(2)] => [0, 1] => [-1, 1]
        _sign_log_x = sign * torch.log(_shift_x) / torch.log(torch.tensor(2.))
        # [-1, 1] => [-n, n] => {-n ,n} => {-n, n-1}
        _quantized_bit = torch.clamp(torch.round(_sign_log_x * n), -n, n - 1)
        # {-n, n} => {-1, 1} => {0, 1}
        _abs_quantized_log_x = torch.abs(_quantized_bit) / n
        # {0, 1} => {0, log(2)} => {1, 2} => {0, 1} => {-1, 1}
        _quantized_normalized_x = sign * (torch.exp(_abs_quantized_log_x * torch.log(torch.tensor(2.))) - 1)

        return _quantized_normalized_x, _quantized_bit

    @staticmethod
    def backward(ctx, _grad_quantized_symmetric_normalized_x, _grad_quantized_bit):
        return _grad_quantized_symmetric_normalized_x, None


class Function_log_sqrt_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _symmetric_normalized_x, _bit):
        ctx.save_for_backward(_symmetric_normalized_x)

        sign = torch.sign(_symmetric_normalized_x)
        # [-1, 1] => [0, 1]
        _abs_x = torch.abs(_symmetric_normalized_x)
        # [0, 1] => [1, 2]
        _shift_x = _abs_x + 1
        # [1, 2] => [0, log(2)] => [0, 1]
        _log_x = torch.log(_shift_x) / torch.log(torch.tensor(2.))
        # [0, 1] => [0, 1]
        _sqrt_x = torch.sqrt(_log_x)
        # [0, 1] => {0, 1}
        _quantized_sqrt_x, _quantized_bit = direct_biased_quantize(_sqrt_x, _bit)
        # {0, 1} => {0, 1} {0, log(2)} => {1, 2} => {0, 1} => {-1, 1}
        _quantized_symmetric_normalized_x = (torch.exp(torch.pow(_quantized_sqrt_x, 2) * torch.log(torch.tensor(2.))) - 1) * sign

        return _quantized_symmetric_normalized_x, _quantized_bit

    @staticmethod
    def backward(ctx, _grad_quantized_symmetric_normalized_x, _grad_quantized_bit):
        return _grad_quantized_symmetric_normalized_x, None


class Function_log10_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _symmetric_normalized_x, _bit):
        ctx.save_for_backward(_symmetric_normalized_x)

        sign = torch.sign(_symmetric_normalized_x)
        # [-1, 1] => [0, 1]
        _abs_x = torch.abs(_symmetric_normalized_x)
        # [0, 1] => [1, 2]
        _shift_x = _abs_x + 1
        # [1, 2] => [0, log10(2)] => [0, 1]
        _log_x = torch.log10(_shift_x) / torch.log10(torch.tensor(2.))
        # [0, 1] => {0, 1}
        _quantized_log_x, _quantized_bit = direct_biased_quantize(_log_x, _bit)
        # {0, 1} => {0, log10(2)} => {1, 2} => {0, 1} => {-1, 1}
        _quantized_symmetric_normalized_x = (10 ** (_quantized_log_x * torch.log2(torch.tensor(2.))) - 1) * sign

        return _quantized_symmetric_normalized_x, _quantized_bit

    @staticmethod
    def backward(ctx, _grad_quantized_symmetric_normalized_x, _grad_quantized_bit):
        return _grad_quantized_symmetric_normalized_x, None


class Function_gaussian_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _symmetric_x, _bit):

        ctx.save_for_backward(_symmetric_x)
        # """
        n = 2 ** (_bit - 1)
        uni_range = torch.arange(-n, n) / n
        sign = torch.sign(uni_range)
        print(uni_range)
        # log_range = torch.exp(uni_range + 1) / torch.exp(torch.tensor(1.))
        # log_range = torch.cat([-log_range + 1, log_range - 1])
        log_range = sign * (1 - gaussian(uni_range, 0, 1) / gaussian(torch.zeros([]), 0, 1)) / gaussian(torch.ones([]), 0, 1)
        print(log_range)
        # """

        _symmetric_quantized_x, _quantized_bit = project(_symmetric_x, log_range, 2 ** _bit)

        return _symmetric_quantized_x, _quantized_bit

    @staticmethod
    def backward(ctx, _grad_symmetric_quantized_x, _grad_quantized_bit):
        return _grad_symmetric_quantized_x, None


def project(x, available_set, n):

    index = torch.argmin(torch.abs(x.view(-1, 1) - available_set.view(1, -1)), dim=1)
    projected_x = torch.matmul(
        torch.nn.functional.one_hot(index.type(torch.LongTensor), n).type(torch.FloatTensor),
        available_set.view(-1, 1)
    )
    return projected_x, index.view(-1)


class Function_biased_gradient_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _input, _bit, _left_threshold, _right_threshold):
        ctx.save_for_backward(_input, _bit, _left_threshold, _right_threshold)
        return _input

    @staticmethod
    def backward(ctx, _grad_output):
        _input, _bit, _left_threshold, _right_threshold = ctx.saved_tensors
        _clipped_grad_output = torch.clip(_grad_output, _left_threshold, _right_threshold)
        _shifted_grad_output = (_clipped_grad_output - _left_threshold) / (_right_threshold - _left_threshold)

        if _bit < 0 or _bit == 32:
            _normalized_quantized_grad_output, _quantized_bit_grad_output = direct_biased_quantize(_shifted_grad_output, 8)
        else:
            _normalized_quantized_grad_output, _quantized_bit_grad_output = direct_biased_quantize(_shifted_grad_output, _bit)

        _quantized_grad_output = _normalized_quantized_grad_output * (_right_threshold - _left_threshold) + _left_threshold

        return _quantized_grad_output, None, None, None


class Function_unbiased_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _normalized_weight, _bit):
        """
        Args:
            ctx:
            _normalized_weight: weights within [-1, 1]
            _bit:
        Returns:
        """
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


class Function_unbiased_quantize_STE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _normalized_weight, _bit, _mask):
        ctx.save_for_backward(_mask)
        n = 2 ** (_bit - 1)
        _round_x = torch.round(_normalized_weight * n)  # [-1, 1] => [-n, n]
        # {-n, n, 1} => {-n, n-1, 1}
        _quantized_bit = torch.clip(
            _round_x, -n, n - 1
        )
        return _quantized_bit / n, _quantized_bit

    @staticmethod
    def backward(ctx, _grad_normalized_quantized_weight, _grad_quantized_bit):
        _mask, = ctx.saved_tensors
        return _grad_normalized_quantized_weight * _mask, None, None



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

    import torch
    import matplotlib.pyplot as plt

    # plt.style.use('ggplot')

    N = 1000
    bins = 100
    n_trial = 10
    bit = 4
    n = 2 ** bit
    n_loss_list = [[], [], []]
    # """
    for _ in range(n_trial):

        inputs = torch.normal(mean=torch.zeros(N), std=torch.ones(N))
        _min, _max = inputs.min(), inputs.max()
        _max_abs = torch.max(-_min, _max)
        _mid = (_min + _max) / 2

        normalized_asymmetric_inputs = (inputs - _min) / (_max - _min)
        quantized_normalized_uni_outputs, quantized_uni_bits = Function_biased_quantize.apply(normalized_asymmetric_inputs, bit)
        quantized_uni_outputs = quantized_normalized_uni_outputs * (_max - _min) + _min

        normalized_symmetric_inputs = (inputs - _mid) / (_max_abs - _mid)

        quantized_normalized_log_outputs, quantized_log_bits = Function_log_quantize.apply(normalized_symmetric_inputs, bit)
        quantized_log_outputs = quantized_normalized_log_outputs * (_max_abs - _mid) + _mid

        # quantized_normalized_log10_outputs, quantized_log10_bits = Function_log10_quantize.apply(normalized_symmetric_inputs, bit)
        # quantized_log10_outputs = quantized_normalized_log10_outputs * (_max_abs - _mid) + _mid

        quantized_normalized_log_sqrt_outputs, quantized_log_sqrt_bits = Function_log_sqrt_quantize.apply(normalized_symmetric_inputs, bit)
        quantized_log_sqrt_outputs = quantized_normalized_log_sqrt_outputs * (_max_abs - _mid) + _mid

        # plt.figure()
        # plt.subplot(4, 3, 1)
        # plt.hist(inputs.numpy(), bins=bins)
        # plt.subplot(4, 3, 2)
        # plt.hist(normalized_asymmetric_inputs.numpy(), bins=bins)
        # plt.subplot(4, 3, 3)
        # plt.hist(normalized_symmetric_inputs.numpy(), bins=bins)
        #
        # plt.subplot(4, 3, 4)
        # plt.hist(quantized_normalized_uni_outputs.numpy(), bins=bins)
        # plt.subplot(4, 3, 5)
        # plt.hist(quantized_uni_bits.numpy(), bins=bins)
        # plt.subplot(4, 3, 6)
        # plt.hist(quantized_uni_outputs.numpy(), bins=bins)
        #
        # plt.subplot(4, 3, 7)
        # plt.hist(quantized_normalized_log_outputs.numpy(), bins=bins)
        # plt.subplot(4, 3, 8)
        # plt.hist(quantized_log_bits.numpy(), bins=bins)
        # plt.subplot(4, 3, 9)
        # plt.hist(quantized_log_outputs.numpy(), bins=bins)
        #
        # plt.subplot(4, 3, 10)
        # plt.hist(quantized_normalized_log_sqrt_outputs.numpy(), bins=bins)
        # plt.subplot(4, 3, 11)
        # plt.hist(quantized_log_sqrt_bits.numpy(), bins=bins)
        # plt.subplot(4, 3, 12)
        # plt.hist(quantized_log_sqrt_outputs.numpy(), bins=bins)
        # plt.show()

        n_loss_list[0].append(torch.mean(torch.abs(quantized_uni_outputs - inputs)).item())
        n_loss_list[1].append(torch.mean(torch.abs(quantized_log_outputs - inputs)).item())
        # n_loss_list[2].append(torch.mean(torch.abs(quantized_log10_outputs - inputs)).item())
        n_loss_list[2].append(torch.mean(torch.abs(quantized_log_sqrt_outputs - inputs)).item())


    for n_loss in n_loss_list:
        print("%.4e(%.4e)" % (np.mean(n_loss), np.std(n_loss)))
    # """

    """
    inputs = torch.normal(mean=torch.zeros(N), std=torch.ones(N))
    # inputs = torch.exp(torch.rand([N]))
    _min, _max = inputs.min(), inputs.max()
    _sign = torch.sign(inputs)
    normalized_inputs = (inputs - _min) / (_max - _min)
    log_outputs, log_outputs_bits, log_outputs_log = Function_biased_log_quantize.apply(normalized_inputs, bit)
    plt.subplot(1, 3, 1)
    plt.hist(normalized_inputs.numpy(), bins=bins)
    plt.subplot(1, 3, 2)
    plt.hist(log_outputs.numpy(), bins=bins)
    plt.subplot(1, 3, 3)
    plt.hist(log_outputs_log.numpy(), bins=bins)
    plt.show()
    """

    # The following test the functionality of project
    """
    inputs = torch.normal(mean=torch.zeros(N), std=torch.ones(N))
    quantized_points = torch.rand([n])
    projected_input = project(inputs, quantized_points, n)
    plt.subplot(1, 2, 1)
    plt.hist(inputs.numpy(), bins=bins)
    plt.subplot(1, 2, 2)
    plt.hist(projected_input.numpy(), bins=bins)
    plt.show()
    """

    # The following test the functionalty of Function_log_quantize
    """
    inputs = torch.normal(mean=torch.zeros(N), std=torch.ones(N))
    normalized_inputs = inputs / torch.max(torch.abs(inputs))
    quantized_inputs, quantized_bits = Function_log_quantize.apply(normalized_inputs, bit)
    plt.subplot(1, 3, 1)
    plt.hist(normalized_inputs.numpy(), bins=bins)
    plt.subplot(1, 3, 2)
    plt.hist(quantized_inputs.numpy(), bins=bins)
    plt.subplot(1, 3, 3)
    plt.hist(quantized_bits.numpy(), bins=bins)
    plt.show()
    """