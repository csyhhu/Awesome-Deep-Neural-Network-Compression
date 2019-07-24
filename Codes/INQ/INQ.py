"""
Experimental implementation of Incremental Network Quantization
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import get_dataloader, get_lmdb_imagenet
from models.INQ_resnet import resnet20_cifar, resnet20_stl
from utils.INQ import partition_quantize_weight, check_INQ_bits
from utils.train import accuracy, test
from utils.recorder import Recorder
from utils.miscellaneous import get_layer

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Approximation Training')
parser.add_argument('--model', '-m', type=str, default='ResNet20', help='Model Arch')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--optimizer', '-o', type=str, default='Adam', help='Optimizer Method')
parser.add_argument('--bitW', '-bw', type=int, default=1, help='Number of quantization bits')
parser.add_argument('--quantize_portion','-q', nargs='+', help='Variation of Quantization Portion', required=True)
parser.add_argument('--exp_spec', '-e', type=str, default='', help='Experiment Specification')
parser.add_argument('--init_lr', '-lr', type=float, default=1e-3, help='Initial Learning rate')
parser.add_argument('--n_epoch', '-n', type=int, default=100, help='Maximum training epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size')
args = parser.parse_args()

# ------------------------------------------
use_cuda = torch.cuda.is_available()
model_name = args.model
dataset_name = args.dataset
quantize_portion_list = args.quantize_portion
MAX_EPOCH = args.n_epoch
optimizer_type = args.optimizer # ['SGD', 'SGD-M', 'adam']
dataset_type = 'large' if dataset_name in ['ImageNet'] else 'small'
batch_size = args.batch_size
bitW = args.bitW
# ------------------------------------------

print(args)
input('Take a look')

# Integerize the quantization portion
for idx, quantize_portion in enumerate(quantize_portion_list):
    quantize_portion_list[idx] = int(quantize_portion)

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
        net = resnet20_cifar(num_classes=num_classes)
    elif dataset_name in ['STL10']:
        net = resnet20_stl(num_classes=num_classes)
    else:
        raise NotImplementedError
    pretrain_path = '%s/%s-%s-pretrain.pth' % (save_root, model_name, dataset_name)
else:
    raise NotImplementedError

net.load_state_dict(torch.load(pretrain_path), strict=False)

if use_cuda:
    net.cuda()

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
SummaryPath = '%s/runs-INQ/%s-%dbits' %(save_root, optimizer_type, bitW)
for quantize_portion in quantize_portion_list:
    SummaryPath += ('-%d' %quantize_portion)
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

################################
# Initial stopGradientMaskDict #
################################
stopGradientMaskDict = dict()
n1_dict = dict()
for layer_name, layer_info  in net.layer_name_list:
    layer_weight = get_layer(net, layer_info).weight.data
    stopGradientMask = torch.ones(layer_weight.shape)
    if use_cuda:
        stopGradientMask = stopGradientMask.cuda()
    stopGradientMaskDict[layer_name] = stopGradientMask
    n1_dict[layer_name] = torch.floor(torch.log2(4 * torch.max(torch.abs(layer_weight)) / 3))


#############
# Begin INQ #
#############
for idx, quantize_portion in enumerate(quantize_portion_list):

    print('%d-th incremental quantization with quantization portion as: %d' %(idx, quantize_portion))

    # Quantize weights and generate new stopGradientMaskDict
    for layer_name, layer_info in net.layer_name_list:
        layer_weight = get_layer(net, layer_info).weight
        quantWeight, new_stopGradientMask = \
            partition_quantize_weight(layer_weight.data, stopGradientMaskDict[layer_name],
                                      n1=n1_dict[layer_name], quant_ratio=quantize_portion, bitW=bitW)
        layer_weight.data.copy_(quantWeight)
        stopGradientMaskDict[layer_name] = new_stopGradientMask

    #########
    # Reset #
    #########
    recorder.smallest_training_loss = 1e9
    recorder.reset_best_test_acc()
    recorder.stop = False

    if optimizer_type == 'SGD-M':
        optimizer = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.init_lr)
    elif optimizer_type in ['adam', 'Adam']:
        optimizer = optim.Adam(net.parameters(), lr=args.init_lr)
    else:
        raise NotImplementedError

    ###############################
    # Train non-quantized weights #
    ###############################
    for epoch in range(MAX_EPOCH):

        if recorder.stop:
            break

        net.train()
        end = time.time()

        recorder.reset_performance()

        print('Epoch: %d, lr: %e' %(epoch, optimizer.param_groups[0]['lr']))

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()

            outputs = net(inputs, stopGradientMaskDict=stopGradientMaskDict)
            losses = nn.CrossEntropyLoss()(outputs, targets)
            losses.backward()

            optimizer.step()

            recorder.update(loss=losses.item(), acc=accuracy(outputs.data, targets.data, (1, 5)),
                            batch_size=outputs.shape[0], cur_lr=optimizer.param_groups[0]['lr'], end=end)

            recorder.print_training_result(batch_idx, len(train_loader))
            end = time.time()

        test_acc = test(net, test_loader=test_loader, dataset_name=dataset_name)

        recorder.update(loss=None, acc=test_acc, batch_size=0, end=None, is_train=False)
        # Adjust lr
        recorder.adjust_lr(optimizer=optimizer)

    print('Best test acc: %s in quantization period %d' \
          %(recorder.get_best_test_acc(), idx))

print('Check INQ bits')
check_INQ_bits(net)
# print(net.conv1.weight.data[0,0,:,:])

recorder.close()