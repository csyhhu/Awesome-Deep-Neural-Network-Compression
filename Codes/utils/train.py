import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import os
import time
import sys
from datetime import datetime
import numpy as np

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 15.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    # L.append('  Step: %s' % format_time(step_time))
    # L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / float(self.count)


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def adjust_learning_rate(optimizer, epoch, init_lr=1e-3, decrease_epoch=5):
    """Sets the learning rate to the initial LR decayed by 10 every decrease_epoch epochs"""
    lr = init_lr * (0.1 ** (epoch // decrease_epoch))
    print ('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(_net, _train_loader, _optimizer, _criterion, _device = 'cpu', _recorder = None):

    _net.train()
    _train_loss = 0
    _correct = 0
    _total = 0

    for batch_idx, (inputs, targets) in enumerate(_train_loader):

        inputs, targets = inputs.to(_device), targets.to(_device)

        _optimizer.zero_grad()
        outputs = _net(inputs)
        losses = _criterion(outputs, targets)
        losses.backward()
        _optimizer.step()

        _train_loss += losses.data.item()
        _, predicted = torch.max(outputs.data, 1)
        _total += targets.size(0)
        _correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(
            batch_idx, len(_train_loader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (_train_loss / (batch_idx + 1), 100. * _correct / _total, _correct, _total)
        )

        if _recorder is not None:
            _recorder.update(loss=losses.data.item(), acc=[_correct / _total], batch_size=inputs.size(0), is_train=True)


    return _train_loss / (len(_train_loader)), _correct / _total


def test(_net, _test_loader, _criterion, _device = 'cpu', _recorder = None):

    _net.eval()

    _test_loss = 0
    _correct = 0
    _total = 0

    for batch_idx, (inputs, targets) in enumerate(_test_loader):

        inputs, targets = inputs.to(_device), targets.to(_device)

        with torch.no_grad():
            outputs = _net(inputs)

        losses = _criterion(outputs, targets)

        _test_loss += losses.data.item()
        _, predicted = torch.max(outputs.data, 1)
        _total += targets.size(0)
        _correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(
            batch_idx,
            len(_test_loader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (_test_loss / (batch_idx + 1), 100. * float(_correct) / _total, _correct, _total)
        )

    if _recorder is not None:
        _recorder.update(loss=_test_loss, acc=[float(_correct) / _total], is_train=False)

    return _test_loss / (len(_test_loader)), float(_correct) / _total