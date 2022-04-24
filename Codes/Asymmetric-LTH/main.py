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

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', '-m', type=str, default='ResNet20', help='Model arch')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--optimizer', '-o', type=str, default='SGD', help='Optimizer')
parser.add_argument('--max_epoch', '-epoch', default=30, type=int, help='Number of maximum epoch')
parser.add_argument('--lr_adjust', '-ad', default=10, type=int, help='Training strategy')
parser.add_argument('--bitW', '-bw', default=4, type=int, help='Quantization bit for weight')
parser.add_argument('--bitA', '-ba', default=4, type=int, help='Quantization bit for input')
parser.add_argument('--bitG', '-bg', default=8, type=int, help='Quantization bit for gradient')
parser.add_argument('--out_of_domain_kernel', '-pk', default=1., type=float, help='Out-of-domain penalty in kernel')
parser.add_argument('--out_of_domain_activation', '-pa', default=1., type=float, help='Out-of-domain penalty in activation')
parser.add_argument('--out_of_domain_gradient', '-pg', default=1., type=float, help='Out-of-domain penalty in gradient')
parser.add_argument('--gradient_quantization_type', '-gq', default='alth', type=str, help='Gradient Quantization Type')
parser.add_argument('--retrain', '-retrain', action='store_true', help='Whether to retrain from a pre-trained model')
parser.add_argument('--retrain_ckpt_path', '-retrain_ckpt', default=None, help='Path to pre-trained model')
parser.add_argument('--pretrain', '-pretrain',  action='store_true', help='Whether to use a pre-trained model')
parser.add_argument('--pretrain_path', '-pretrain_path', default=None, help='Path to pre-trained model')
parser.add_argument('--exp_spec', '-exp', default=None, help='Experiment specification')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
model_name = args.model
dataset_name = args.dataset

# Suppose I want to retrain from another ckpt
ckpt_root = './Results/%s-%s/asymmetric-LTH/checkpoint/' % (model_name, dataset_name)
if not os.path.exists(ckpt_root):
    os.makedirs(ckpt_root)
ckpt_path = os.path.join(
    ckpt_root, '%s-bitW-%d-%.0e-bitA-%d-%.0e-%s-lr-adjust-%d-epoch-%d%s%s%s.ckpt' % (
        args.optimizer,
        args.bitW, args.out_of_domain_kernel,
        args.bitA, args.out_of_domain_activation,
        'bitG-%d-%.0e' % (args.bitG, args.out_of_domain_gradient) if args.gradient_quantization_type == 'alth' else 'bitG-%d-maq' % (args.bitG),
        args.lr_adjust, args.max_epoch,
        "-pretrain" if args.pretrain else "", "-retrain" if args.retrain else "",
        '-%s' % args.exp_spec if args.exp_spec is not None else ''
    )
)

save_root = './Results/%s-%s/asymmetric-LTH/%s-bitW-%d-%.0e-bitA-%d-%.0e-%s-lr-adjust-%d-epoch-%d%s%s%s' %(
    model_name, dataset_name, args.optimizer,
    args.bitW, args.out_of_domain_kernel,
    args.bitA, args.out_of_domain_activation,
    'bitG-%d-%.0e' % (args.bitG, args.out_of_domain_gradient) if args.gradient_quantization_type == 'alth' else 'bitG-%d-maq' % (args.bitG),
    args.lr_adjust, args.max_epoch,
    "-pretrain" if args.pretrain else "", "-retrain" if args.retrain else "",
    '-%s' % args.exp_spec if args.exp_spec is not None else ''
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
    net = resnet20_cifar(num_classes=num_classes, bitW=args.bitW, bitA=args.bitA, bitG=args.bitG, out_of_domain_kernel=args.out_of_domain_kernel, out_of_domain_activation=args.out_of_domain_activation, out_of_domain_gradient=args.out_of_domain_gradient, gradient_quantized_type=args.gradient_quantization_type)
elif model_name == 'ResNet32':
    net = resnet20_cifar(num_classes=num_classes, bitW=args.bitW, bitA=args.bitA)
elif model_name == 'ResNet56':
    net = resnet20_cifar(num_classes=num_classes, bitW=args.bitW, bitA=args.bitA)
else:
    raise NotImplementedError

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if args.optimizer in ['Adam', 'adam']:
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if args.retrain:
    start_epoch, best_test_acc = load_checkpoint(net, optimizer, args.retrain_ckpt_path)
    if args.lr != optimizer.param_groups[0]['lr']:
        optimizer.param_groups[0]['lr'] = args.lr
        print('Change learning rate to %.3e' % optimizer.param_groups[0]['lr'])
else:
    start_epoch = 0
    best_test_acc = 0

if args.pretrain:
    print('Load pretrained model from %s' % (args.pretrain_path))
    net.load_state_dict(torch.load(args.pretrain_path), strict=False)

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
writer = SummaryWriter(save_root)
# Initialize recorder for threshold
"""
weight_threshold_recorder_collections = {}
input_threshold_recorder_collections = {}
weight_quantization_error_recorder_collection = {}
input_quantization_error_recorder_collection = {}
gradient_threshold_recorder_collection = {}
gradient_out_of_domain_collection = {}
gradient_quantization_error_collection = {}
min_max_weight_recorder_collection = {}
min_max_input_recorder_collection = {}
weight_bit_allocation_collection = {}
input_bit_allocation_collection = {}
for name, layer in net.quantized_layer_collections.items():
    if not os.path.exists('%s/%s' % (save_root, name)):
        os.makedirs('%s/%s' % (save_root, name))
    weight_threshold_recorder_collections[name] = open('%s/%s/weight_threshold.txt' % (save_root, name), 'a+')
    input_threshold_recorder_collections[name] = open('%s/%s/input_threshold.txt' % (save_root, name), 'a+')
    gradient_threshold_recorder_collection[name] = open('%s/%s/gradient_threshold.txt' % (save_root, name), 'a+')
    weight_quantization_error_recorder_collection[name] = open('%s/%s/weight_quantization_error.txt' % (save_root, name), 'a+')
    input_quantization_error_recorder_collection[name] = open('%s/%s/input_quantization_error.txt' % (save_root, name), 'a+')
    min_max_weight_recorder_collection[name] = open('%s/%s/min_max_weight.txt' % (save_root, name), 'a+')
    min_max_input_recorder_collection[name] = open('%s/%s/min_max_input.txt' % (save_root, name), 'a+')
    weight_bit_allocation_collection[name] = open('%s/%s/weight_bit_allocation.txt' % (save_root, name), 'a+')
    input_bit_allocation_collection[name] = open('%s/%s/input_bit_allocation.txt' % (save_root, name), 'a+')
    gradient_out_of_domain_collection[name] = open('%s/%s/gradient_out_of_domain.txt' % (save_root, name), 'a+')
    gradient_quantization_error_collection[name] = open('%s/%s/gradient_quantization_error.txt' % (save_root, name), 'a+')
"""

for epoch in range(start_epoch, start_epoch + max_training_epoch):

    print('Epoch: [%3d]' % epoch)
    train_loss, train_acc = train(
        net, train_loader, optimizer, criterion, _device=device,
        _writer=writer,
        _recorder=recorder,
        # _weight_threshold_recorder_collection=weight_threshold_recorder_collections,
        # _input_threshold_recorder_collection=input_threshold_recorder_collections,
        # _gradient_threshold_recorder_collection=gradient_threshold_recorder_collection,
        # _weight_quantization_error_collection=weight_quantization_error_recorder_collection,
        # _input_quantization_error_collection=input_quantization_error_recorder_collection,
        # _min_max_weight_collection=min_max_weight_recorder_collection,
        # _min_max_input_collection=min_max_input_recorder_collection,
        # _weight_bit_allocation_collection=weight_bit_allocation_collection,
        # _input_bit_allocation_collection=input_bit_allocation_collection,
        # _gradient_out_of_domain_collection=gradient_out_of_domain_collection,
        _n_batch_used=-1,
        _epoch_idx=epoch
    )
    # """
    print('\nTest\n')
    test_loss, test_acc = test(
        net, test_loader, criterion, _device=device, _recorder=recorder
    )
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        save_checkpoint(net, optimizer, epoch, best_test_acc, ckpt_path)

    if writer is not None:
        writer.add_scalar("test/acc", test_acc, epoch)

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
    # """
recorder.close()
writer.close()
"""
for collection in [
    weight_threshold_recorder_collections, input_threshold_recorder_collections, gradient_threshold_recorder_collection,
    weight_quantization_error_recorder_collection, input_quantization_error_recorder_collection,
    min_max_weight_recorder_collection, min_max_input_recorder_collection,
    weight_bit_allocation_collection, input_bit_allocation_collection,
    gradient_out_of_domain_collection, gradient_quantization_error_collection
]:
    for recorder in collection.values():
        recorder.close()
"""