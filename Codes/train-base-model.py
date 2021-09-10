'''Train CIFAR10 with PyTorch.'''

"""
This code is forked and modified from 'https://github.com/kuangliu/pytorch-cifar'. Thanks to its contribution.

A 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse

from models.CIFARNet import CIFARNet
from models.resnet import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar
from torch.autograd import Variable
from utils.train import progress_bar, is_int, train, test
from utils.dataset import get_dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--retrain', '-r', default=None, help='Retrain from a pre-trained model')
parser.add_argument('--model', '-m', type=str, default='CIFARNet', help='Model arch')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--optimizer', '-o', type=str, default='SGD', help='Optimizer')
parser.add_argument('--adjust', '-ad', default=30, type=int, help='Training strategy')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
model_name = args.model
dataset_name = args.dataset

save_root = './Results/%s-%s' %(model_name, dataset_name)
if not os.path.exists(save_root):
    os.makedirs(save_root)

# Data
print('==> Preparing data..')

train_loader = get_dataloader(dataset_name, 'train', 128)
test_loader = get_dataloader(dataset_name, 'test', 100)

if dataset_name in ['CIFAR10', 'STL10']:
    num_classes = 10
elif dataset_name in ['CIFAR100']:
    num_classes = 100
else:
    raise ValueError('Dataset %s not supported.' % dataset_name)

# Model
print('==> Building model..')
if model_name == 'ResNet20':
    net = resnet20_cifar(num_classes=num_classes)
elif model_name == 'ResNet32':
    net = resnet20_cifar(num_classes=num_classes)
elif model_name == 'ResNet56':
    net = resnet20_cifar(num_classes=num_classes)
else:
    raise NotImplementedError

if args.retrain is not None:
    # Load pretrain ckpt.
    print('==> Retrain from pre-trained model %s' % args.retrain)
    ckpt = torch.load('%s/checkpoint/%s_ckpt.t7' % (save_root, model_name))
    net = ckpt['net']
    start_epoch = ckpt['epoch']
    best_test_acc = ckpt['test_acc']
else:
    start_epoch = 0
    best_test_acc = 0

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

if args.optimizer in ['Adam', 'adam']:
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Begin Training
ascent_count = 0
min_train_loss = 1e9
max_training_epoch = args.ad if is_int(args.ad) else 200

for epoch in range(start_epoch, start_epoch + max_training_epoch):

    print('Epoch: [%3d]' % epoch)
    train_loss, train_acc = train(net, train_loader, optimizer, criterion, device)
    test_loss, test_acc = test(net, test_loader, criterion, device)

    # Save checkpoint.
    if test_acc > best_test_acc:
        print('Saving...')
        state = {
            'net': net.module if use_cuda else net,
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('%s/checkpoint' % (save_root)):
            os.mkdir('%s/checkpoint' % (save_root))
        torch.save(state, '%s/checkpoint/%s_ckpt.t7' % (save_root, model_name))
        best_test_acc = test_acc
        torch.save(net.module.state_dict(), '%s/%s-%s-pretrain.pth' % (save_root, model_name, dataset_name))

    if args.ad == 'adaptive':

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            ascent_count = 0
        else:
            ascent_count += 1

        print('Current Loss: %.3f [%.3f], ascent count: %d' % (train_loss, min_train_loss, ascent_count))

        if ascent_count >= 3:
            optimizer.param_groups[0]['lr'] *= 0.1
            ascent_count = 0
            if (optimizer.param_groups[0]['lr']) < (args.lr * 1e-3):
                break