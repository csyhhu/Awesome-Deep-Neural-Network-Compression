"""
Experimental implementation of Trained Ternary Quantization
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.dataset import get_dataloader, get_lmdb_imagenet
from models.ttq_resnet import resnet20_cifar, resnet20_stl
from utils.TTQ import test, measure_net_stats
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
parser.add_argument('--init_lr', '-lr', type=float, default=1e-3, help='Initial Learning rate')
parser.add_argument('--n_epoch', '-n', type=int, default=100, help='Maximum training epochs')
parser.add_argument('--lr_adjust', '-ad', type=str, default='dorefa', help='LR adjusting method')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size')
parser.add_argument('--thresh_factor', '-tf', type=float, default=0.05, help='Batch size')
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
thresh_factor = args.thresh_factor
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
    if dataset_name in ['CIFAR10', 'CIFAR100']:
        net = resnet20_cifar(num_classes=num_classes, thresh_factor=thresh_factor)
    else:
        net = resnet20_stl(num_classes=num_classes, thresh_factor=thresh_factor)
    pretrain_path = '%s/%s-%s-pretrain.pth' % (save_root, model_name, dataset_name)
else:
    raise NotImplementedError

net.load_state_dict(torch.load(pretrain_path), strict=False)

if use_cuda:
    net.cuda()

if optimizer_type == 'SGD-M':
    optimizer = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=5e-4)
elif optimizer_type == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.init_lr)
elif optimizer_type in ['adam', 'Adam']:
    optimizer = optim.Adam(net.parameters(), lr=args.init_lr)
else:
    raise NotImplementedError


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
SummaryPath = '%s/runs-TTQ/TheshFactor-%.3f-Optimizer-%s' \
               %(save_root, thresh_factor, optimizer_type)
if args.exp_spec is not '':
    SummaryPath += ('-' + args.exp_spec)

print('Save to %s' %SummaryPath)

if os.path.exists(SummaryPath):
    print('Record exist, remove')
    input()
    shutil.rmtree(SummaryPath)
    os.makedirs(SummaryPath)
else:
    os.makedirs(SummaryPath)

recorder = Recorder(SummaryPath=SummaryPath, dataset_name=dataset_name)
conv1_pos_file = open('%s/conv1_pos.txt' %SummaryPath, 'w+')
conv1_neg_file = open('%s/conv1_neg.txt' %SummaryPath, 'w+')
conv1_prune_file = open('%s/conv1_prune.txt' %SummaryPath, 'w+')

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

        outputs = net(inputs)
        losses = nn.CrossEntropyLoss()(outputs, targets)
        losses.backward()

        optimizer.step()

        ####################################
        # Measure the Variation of pos/neg #
        ####################################
        conv1_pos, conv1_neg, conv1_pos_rate, conv1_neg_rate, conv1_prune_rate = measure_net_stats(net.conv1)
        conv1_pos_file.write('%d, %.3f, %.3f\n' %(batch_idx + epoch * len(train_loader), conv1_pos, conv1_pos_rate))
        conv1_neg_file.write('%d, %.3f, %.3f\n' % (batch_idx + epoch * len(train_loader), conv1_neg, conv1_neg_rate))
        conv1_prune_file.write('%d, %.3f\n' %(batch_idx + epoch * len(train_loader), conv1_prune_rate))
        conv1_pos_file.flush()
        conv1_neg_file.flush()
        conv1_prune_file.flush()

        recorder.update(loss=losses.item(), acc=accuracy(outputs.data, targets.data, (1, 5)),
                        batch_size=outputs.shape[0], cur_lr=optimizer.param_groups[0]['lr'], end=end)

        recorder.print_training_result(batch_idx, len(train_loader),
                                       append='pos: %.3f, neg: %.3f, pos rate: %.3f, neg rate: %.3f, prune rate: %.3f'
                                              %(conv1_pos, conv1_neg, conv1_pos_rate, conv1_neg_rate, conv1_prune_rate))
        end = time.time()

    test_acc = test(net, test_loader=test_loader, dataset_name=dataset_name)

    recorder.update(loss=None, acc=test_acc, batch_size=0, end=None, is_train=False)
    # Adjust lr
    recorder.adjust_lr(optimizer=optimizer)

print('Best test acc: %s' %recorder.get_best_test_acc())
recorder.close()
conv1_pos_file.close()
conv1_neg_file.close()
conv1_prune_file.close()