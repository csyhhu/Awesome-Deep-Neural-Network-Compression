'''Train CIFAR10 with PyTorch.'''

"""
This code is forked and modified from 'https://github.com/kuangliu/pytorch-cifar'. Thanks to its contribution.

A 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import argparse

from resnet import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar
from utils.train import progress_bar, is_int, test
from train import train
from utils.dataset import get_dataloader
from utils.recorder import Recorder
from utils.miscellaneous import save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', '-m', type=str, default='ResNet20', help='Model arch')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--optimizer', '-o', type=str, default='SGD', help='Optimizer')
parser.add_argument('--max_epoch', '-epoch', default=30, type=int, help='Number of maximum epoch')
parser.add_argument('--lr_adjust', '-ad', default=10, type=int, help='Training strategy')
parser.add_argument('--bitW', '-bw', default=4, type=int, help='Quantization bit for weight')
parser.add_argument('--bitA', '-ba', default=4, type=int, help='Quantization bit for input')
parser.add_argument('--retrain', '-r', action='store_true', help='Whether to retrain from a pre-trained model')
parser.add_argument('--ckpt_path', '-ckpt', default=None, help='Path to pre-trained model')
parser.add_argument('--pretrain', '-pretrain',  action='store_true', help='Whether to use a pre-trained model')
parser.add_argument('--pretrain_path', '-pretrain_path', default=None, help='Path to pre-trained model')
parser.add_argument('--exp_spec', '-exp', default=None, help='Experiment specification')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
model_name = args.model
dataset_name = args.dataset

if args.ckpt_path is None:
    ckpt_root = './Results/%s-%s/asymmetric-LTH/checkpoint/' % (model_name, dataset_name)
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
    ckpt_path = os.path.join(
        ckpt_root, '%s-bitW-%d-bitA-%d-lr-adjust-%d-epoch-%d%s.ckpt' % (
            args.optimizer, args.bitW, args.bitA, args.lr_adjust, args.max_epoch, '-%s' % args.exp_spec if args.exp_spec is not None else ''
        )
    )
else:
    ckpt_path = args.ckpt_path

save_root = './Results/%s-%s/asymmetric-LTH/%s-bitW-%d-bitA-%d-lr-adjust-%d-epoch-%d%s%s' %(
    model_name, dataset_name, args.optimizer, args.bitW, args.bitA, args.lr_adjust, args.max_epoch,
    "-pretrain" if args.pretrain else "", '-%s' % args.exp_spec if args.exp_spec is not None else ''
)

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
    net = resnet20_cifar(num_classes=num_classes, bitW=args.bitW, bitA=args.bitA)
elif model_name == 'ResNet32':
    net = resnet20_cifar(num_classes=num_classes, bitW=args.bitW, bitA=args.bitA)
elif model_name == 'ResNet56':
    net = resnet20_cifar(num_classes=num_classes, bitW=args.bitW, bitA=args.bitA)
else:
    raise NotImplementedError

if args.optimizer in ['Adam', 'adam']:
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if args.retrain:
    start_epoch, best_test_acc = load_checkpoint(net, optimizer, ckpt_path)
else:
    start_epoch = 0
    best_test_acc = 0

if args.pretrain:
    load_checkpoint(net, None, args.pretrain_path)

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# ---
# Begin Training
# ---
ascent_count = 0
min_train_loss = 1e9
criterion = nn.CrossEntropyLoss()
max_training_epoch = args.max_epoch

# Initialize recorder for general training
recorder = Recorder(SummaryPath=save_root)
recorder.write_arguments([args])
# Initialize recorder for threshold
weight_threshold_recorder_collections = {}
input_threshold_recorder_collections = {}
weight_quantization_error_recorder_collection = {}
input_quantization_error_recorder_collection = {}
min_max_weight_recorder_collection = {}
min_max_input_recorder_collection = {}
weight_bit_allocation_collection = {}
input_bit_allocation_collection = {}
for name, layer in net.quantized_layer_collections.items():
    if not os.path.exists('%s/%s' % (save_root, name)):
        os.makedirs('%s/%s' % (save_root, name))
    weight_threshold_recorder_collections[name] = open('%s/%s/weight_threshold.txt' % (save_root, name), 'a+')
    input_threshold_recorder_collections[name] = open('%s/%s/input_threshold.txt' % (save_root, name), 'a+')
    weight_quantization_error_recorder_collection[name] = open('%s/%s/weight_quantization_error.txt' % (save_root, name), 'a+')
    input_quantization_error_recorder_collection[name] = open('%s/%s/input_quantization_error.txt' % (save_root, name), 'a+')
    min_max_weight_recorder_collection[name] = open('%s/%s/min_max_weight.txt' % (save_root, name), 'a+')
    min_max_input_recorder_collection[name] = open('%s/%s/min_max_input.txt' % (save_root, name), 'a+')
    weight_bit_allocation_collection[name] = open('%s/%s/weight_bit_allocation.txt' % (save_root, name), 'a+')
    input_bit_allocation_collection[name] = open('%s/%s/input_bit_allocation.txt' % (save_root, name), 'a+')

for epoch in range(start_epoch, start_epoch + max_training_epoch):

    print('Epoch: [%3d]' % epoch)
    train_loss, train_acc = train(
        net, train_loader, optimizer, criterion, _device=device, _recorder=recorder,
        _weight_threshold_recorder_collection=weight_threshold_recorder_collections,
        _input_threshold_recorder_collection=input_threshold_recorder_collections,
        _weight_quantization_error_collection=weight_quantization_error_recorder_collection,
        _input_quantization_error_collection=input_quantization_error_recorder_collection,
        _min_max_weight_collection=min_max_weight_recorder_collection,
        _min_max_input_collection=min_max_input_recorder_collection,
        _weight_bit_allocation_collection=weight_bit_allocation_collection,
        _input_bit_allocation_collection=input_bit_allocation_collection
    )
    test_loss, test_acc = test(
        net, test_loader, criterion, _device=device, _recorder=recorder
    )
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        save_checkpoint(net, optimizer, epoch, best_test_acc, ckpt_path)

    if args.lr_adjust == 'adaptive':

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

    elif (epoch + 1) % args.lr_adjust == 0:

        optimizer.param_groups[0]['lr'] *= 0.1
        if (optimizer.param_groups[0]['lr']) < (args.lr * 1e-3):
            break

        print('Learning rate decrease to %e' % optimizer.param_groups[0]['lr'])

recorder.close()
for collection in [
    weight_threshold_recorder_collections, input_threshold_recorder_collections,
    weight_quantization_error_recorder_collection, input_quantization_error_recorder_collection,
    min_max_weight_recorder_collection, min_max_input_recorder_collection,
    weight_bit_allocation_collection, input_bit_allocation_collection
]:
    for recorder in collection.values():
        recorder.close()