"""
Experimental implementation of
Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import get_dataloader, get_lmdb_imagenet
from models.QIL_resnet import resnet20_cifar
from utils.QIL import test
from utils.train import accuracy
from utils.recorder import Recorder

# from tensorboardX import SummaryWriter

import argparse
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Approximation Training')
parser.add_argument('--model', '-m', type=str, default='ResNet20', help='Model Arch')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--optimizer', '-o', type=str, default='Adam', help='Optimizer Method')
parser.add_argument('--exp_spec', '-e', type=str, default='', help='Experiment Specification')
parser.add_argument('--bitW', '-bw', type=int, default=2, help='Number of weights quantization')
parser.add_argument('--bitA', '-ba', type=int, default=2, help='Number of activation quantization')
parser.add_argument('--init_lr', '-lr', type=float, default=1e-3, help='Initial Learning rate')
parser.add_argument('--n_epoch', '-n', type=int, default=100, help='Maximum training epochs')
parser.add_argument('--lr_adjust', '-ad', type=str, default='dorefa', help='LR adjusting method')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size')
args = parser.parse_args()

# ------------------------------------------
use_cuda = torch.cuda.is_available()
model_name = args.model
dataset_name = args.dataset
MAX_EPOCH = args.n_epoch
optimizer_type = args.optimizer # ['SGD', 'SGD-M', 'adam']
dataset_type = 'large' if dataset_name in ['ImageNet'] else 'small'
lr_adjust = args.lr_adjust
batch_size = args.batch_size
bitW = args.bitW
bitA = args.bitA
# ------------------------------------------

print(args)
input('Take a look')

if dataset_name in ['CIFAR10', 'STL10']:
    num_classes = 10
    save_root = '../Results/%s-%s' % (model_name, dataset_name)
elif dataset_name == 'CIFAR100':
    num_classes = 100
    save_root = '../Results/%s-%s' % (model_name, dataset_name)
elif dataset_name == 'ImageNet':
    save_root = '../Results/%s' % model_name
else:
    raise NotImplementedError


###################
# Initial Network #
###################
if model_name == 'ResNet20':
    net = resnet20_cifar(num_classes=num_classes, bitW=bitW, bitA=bitA)
    pretrain_path = '%s/%s-%s-pretrain.pth' % (save_root, model_name, dataset_name)
else:
    raise NotImplementedError

net.load_state_dict(torch.load(pretrain_path), strict=False)

if use_cuda:
    net.cuda()

# if optimizer_type == 'SGD-M':
#     optimizer = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=5e-4)
# elif optimizer_type == 'SGD':
#     optimizer = optim.SGD(net.parameters(), lr=args.init_lr)
# elif optimizer_type in ['adam', 'Adam']:
#     optimizer = optim.Adam(net.parameters(), lr=args.init_lr)
# else:
#     raise NotImplementedError

param_model = list()
param_QIL = list()
for named, param in net.named_parameters():
    if 'pruning' in named or 'clipping' in named or 'gamma' in named:
        param_QIL.append(param)
    else:
        param_model.append(param)

optimizer = optim.Adam(param_model, lr=args.init_lr)
optimizer_QIL = optim.Adam(param_QIL, lr=1e-3, weight_decay=1)

################
# Load Dataset #
################
if dataset_name == 'ImageNet':
    try:
        train_loader = get_lmdb_imagenet('train', batch_size)
        test_loader = get_lmdb_imagenet('test', 100)
    except:
        train_loader = get_dataloader(dataset_name, 'train', batch_size)
        test_loader = get_dataloader(dataset_name, 'test', 100)
else:
    train_loader = get_dataloader(dataset_name, 'train', batch_size)
    test_loader = get_dataloader(dataset_name, 'test', 100)


####################
# Initial Recorder #
####################
SummaryPath = '%s/runs-QIL/Optimizer-%s-bitW-%d-bitA-%d' \
              %(save_root, optimizer_type, bitW, bitA)
if args.exp_spec is not '':
    SummaryPath += ('-' + args.exp_spec)

print('Save to %s' %SummaryPath)

if os.path.exists(SummaryPath):
    print('Record exist, remove')
    # input()
    shutil.rmtree(SummaryPath)
    os.makedirs(SummaryPath)
else:
    os.makedirs(SummaryPath)

recorder = Recorder(SummaryPath=SummaryPath, dataset_name=dataset_name)

for epoch in range(MAX_EPOCH):

    if recorder.stop:
        break

    net.train()
    end = time.time()

    recorder.reset_performance()

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        optimizer_QIL.zero_grad()

        outputs = net(inputs)
        losses = nn.CrossEntropyLoss()(outputs, targets)
        losses.backward()

        optimizer.step()
        optimizer_QIL.step()

        recorder.update(loss=losses.item(), acc=accuracy(outputs.data, targets.data, (1, 5)),
                        batch_size=outputs.shape[0], cur_lr=optimizer.param_groups[0]['lr'], end=end)

        selected_p = net.layer1[0].conv1.pruning_point.data
        recorder.print_training_result(batch_idx, len(train_loader), append='%e' %(selected_p))
        end = time.time()

        if batch_idx == 1:
            ds

    test_acc = test(net, test_loader=test_loader, dataset_name=dataset_name)

    recorder.update(loss=None, acc=test_acc, batch_size=0, end=None, is_train=False)
    # Adjust lr
    if dataset_name == 'ImageNet':
        if epoch == 20 or epoch == 40 :
            optimizer.param_groups[0]['lr'] *= 0.1
    else:
        recorder.adjust_lr(optimizer)

print('Best test acc: %s' %recorder.get_best_test_acc())
recorder.close()