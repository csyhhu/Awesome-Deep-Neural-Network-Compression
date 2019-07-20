"""
An utils code for loading dataset
"""
import os

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.datasets import ImageFolder as lmdb_ImageFolder
from utils.datasets import transforms as lmdb_transforms

from datetime import datetime
import utils.imagenet_utils as imagenet_utils
import utils.CIFAR10_utils as CIFAR10_utils
import utils.SVHN_utils as SVHN_utils


def get_mean_and_std(dataset, n_channels=3):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(n_channels):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_dataloader(dataset_name, split, batch_size, \
                   add_split = None, shuffle = True, ratio=-1, num_workers=4):

    print ('[%s] Loading %s-%s from %s' %(datetime.now(), split, add_split, dataset_name))

    if dataset_name == 'MNIST':

        data_root_list = []

        for data_root in data_root_list:
            if os.path.exists(data_root):
                print('Found %s in %s' % (dataset_name, data_root))
                break

        normalize = transforms.Normalize((0.1307,), (0.3081,))
        # if split == 'train':
        MNIST_transform =transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               normalize
                           ])
        # else:
        dataset = MNIST_utils.MNIST(root=data_root,
                                    train = True if split =='train' else False, add_split=add_split,
                                    download=False, transform=MNIST_transform, ratio=ratio)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    elif dataset_name == 'SVHN':

        data_root_list = []

        for data_root in data_root_list:
            if os.path.exists(data_root):
                print('Found %s in %s' % (dataset_name, data_root))
                break
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        if split == 'train':
            trainset = SVHN_utils.SVHN(root=data_root, split='train', add_split=add_split, download=True,
                                     transform=transforms.Compose([
                                         # transforms.RandomCrop(32, padding=4),
                                         # transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize
                                         ]), ratio=ratio)
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        elif split == 'test':
            testset = SVHN_utils.SVHN(root=data_root, split='test', download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         normalize
                                         ]))
            loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    elif dataset_name == 'CIFAR10':

        data_root_list = []

        for data_root in data_root_list:
            if os.path.exists(data_root):
                print('Found %s in %s' %(dataset_name, data_root))
                break

        if split == 'train':

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = CIFAR10_utils.CIFAR10(root=data_root, train=True, download=True,
                                                    transform=transform_train, ratio=ratio)
            print ('Number of training instances used: %d' %(len(trainset)))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        elif split == 'test' or split == 'val':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                                   transform=transform_test)
            loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    elif dataset_name == 'CIFAR100':

        data_root_list = []
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        if split == 'train':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            trainset = CIFAR10_utils.CIFAR100(root=data_root, train=True, download=True,
                                                    transform=transform_train, ratio=ratio)
            print ('Number of training instances used: %d' %(len(trainset)))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        elif split == 'test' or split == 'val':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True,
                                                   transform=transform_test)
            loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


    elif dataset_name == 'STL10':

        data_root_list = []
        for data_root in data_root_list:
            if os.path.exists(data_root):
                print('Found STL10 in %s' % data_root)
                break

        if split == 'train':
            loader = torch.utils.data.DataLoader(
                datasets.STL10(
                    root=data_root, split='train', download=True,
                    transform=transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(96),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
                batch_size=batch_size, shuffle=True)

        if split in ['test', 'val']:
            loader = torch.utils.data.DataLoader(
                datasets.STL10(
                    root=data_root, split='test', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
                batch_size=batch_size, shuffle=False)


    elif dataset_name == 'ImageNet':
        data_root_list = []
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break
        traindir = ('../train_imagenet_list.pkl', '../classes.pkl', '../classes-to-idx.pkl','%s/train' %data_root)
        valdir = ('../val_imagenet_list.pkl', '../classes.pkl', '../classes-to-idx.pkl', '%s/val-pytorch' %data_root)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if split == 'train':
            trainDataset = imagenet_utils.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), ratio=ratio)
            print ('Number of training data used: %d' %(len(trainDataset)))
            loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, \
                                                 num_workers = num_workers, pin_memory=True)

        elif split == 'val' or split == 'test':
            valDataset = imagenet_utils.ImageFolder(valdir, transforms.Compose([
		        transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
            loader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=True, \
                                                 num_workers = num_workers, pin_memory=True)

    else:
        raise ('Your dataset is not implemented in this project, please write it by your own')

    print ('[DATA LOADING] Loading from %s-%s-%s finish. Number of images: %d, Number of batches: %d' \
           %(dataset_name, split, add_split, len(loader.dataset), len(loader)))

    return loader


def get_lmdb_imagenet(split, batch_size, shuffle = True, num_workers = None):

    data_root_list = []
    data_exist_flag = False
    for data_root in data_root_list:
        if os.path.exists(data_root):
            data_exist_flag = True
            break

    if data_exist_flag is False:
        raise NotImplementedError

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if split in ['val', 'test']:
        loader = torch.utils.data.DataLoader(
            lmdb_ImageFolder(data_root, lmdb_transforms.Compose([
                lmdb_transforms.ToTensor(),
                lmdb_transforms.CenterCrop(224),
                normalize,
            ]),
            Train=False),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=4 * torch.cuda.device_count() if num_workers is None else num_workers,
            pin_memory=True)

    elif split == 'train':
        loader = torch.utils.data.DataLoader(
            lmdb_ImageFolder(data_root, lmdb_transforms.Compose([
                lmdb_transforms.RandomSizedCrop(224),
                lmdb_transforms.RandomHorizontalFlip(),
                lmdb_transforms.ToTensor(),
                normalize,
            ]),
            Train=True),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=4 * torch.cuda.device_count() if num_workers is None else num_workers,
            pin_memory=True)

    else:
        raise NotImplementedError

    print('[DATA LOADING] Load %s finish, with number of batches as: %d' %(split, len(loader)))
    return loader