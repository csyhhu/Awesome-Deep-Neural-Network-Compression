'''Train CIFAR10 with PyTorch.'''

"""
This code is forked and modified from 'https://github.com/kuangliu/pytorch-cifar'. Thanks to its contribution.
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
from models.resnet import resnet20_cifar
from torch.autograd import Variable
from utils.train import progress_bar
from utils.dataset import get_dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--retrain', '-r', default=None, help='Retrain from a pre-trained model')
parser.add_argument('--model', '-m', type=str, default='CIFARNet', help='Model Arch')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--optimizer', '-o', type=str, default='SGD', help='Optimizer')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
model_name = args.model
dataset_name = args.dataset

save_root = './Results/%s-%s' %(model_name, dataset_name)
if not os.path.exists(save_root):
    os.makedirs(save_root)

# Data
print('==> Preparing data..')

trainloader = get_dataloader(dataset_name, 'train', 128)
testloader = get_dataloader(dataset_name, 'test', 100)

# Model
print('==> Building model..')
if model_name == 'CIFARNet':
    if dataset_name in ['CIFAR10', 'STL10']:
        net = CIFARNet(num_classes=10)
    elif dataset_name in ['CIFAR9', 'STL9']:
        net = CIFARNet(num_classes=9)
    else:
        raise ('%s in %s have not been finished' %(model_name, dataset_name))
elif model_name == 'ResNet20':
    net = resnet20_cifar()
else:
    raise NotImplementedError

if args.retrain is not None:
    # Load pretrain model.
    print('==> Retrain from pre-trained model %s' %args.retrain)
    pretrained = torch.load('./Results/%s' %args.retrain)
    net.load_state_dict(pretrained)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

if args.optimizer in ['Adam', 'adam']:
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d, lr: %e' % (epoch, optimizer.param_groups[0]['lr']))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/(len(trainloader))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*float(correct)/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('%s/checkpoint' %(save_root)):
            os.mkdir('%s/checkpoint' %(save_root))
        torch.save(state, '%s/checkpoint/%s_ckpt.t7' %(save_root, model_name))
        best_acc = acc
        torch.save(net.module.state_dict(), '%s/%s-%s-pretrain.pth' %(save_root, model_name, dataset_name))

ascent_count = 0
min_train_loss = 1e9

for epoch in range(start_epoch, start_epoch+200):

    train_loss = train(epoch)
    test(epoch)

    if train_loss < min_train_loss:
        min_train_loss = train_loss
        ascent_count = 0
    else:
        ascent_count += 1

    print('Current Loss: %.3f [%.3f], ascent count: %d'
          %(train_loss, min_train_loss, ascent_count))

    if ascent_count >= 3:
        optimizer.param_groups[0]['lr'] *= 0.1
        ascent_count = 0
        if (optimizer.param_groups[0]['lr']) < (args.lr * 1e-3):
            break
