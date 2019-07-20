from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from torchvision.datasets.utils import download_url, check_integrity

from operator import itemgetter
from random import shuffle
import torch

class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, split='train', add_split=None,
                 transform=None, target_transform=None, download=False, ratio=-1):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        if split == 'train':

            if ratio != -1:
                assert (add_split is not None)
                sample_length = int(len(self.data) * ratio)
                # random_idx = np.random.randint(len(self.data), size=sample_length)
                random_idx = np.random.choice(len(self.data), size=sample_length, replace=False)
                self.data = self.data[random_idx, :, :, :]
                self.labels = itemgetter(*random_idx)(self.labels)
                print('Random sample train data shape: %s, label number: %d' \
                      % (self.data.shape, len(self.labels)))

            if add_split is not None:
                assert (ratio == -1)
                if os.path.exists(os.path.join(self.root, '%s.pt' %add_split)):
                    self.data, self.labels = torch.load(
                        os.path.join(self.root, '%s.pt' %add_split))
                else:
                    print('%s file not exist, create.' % (add_split))
                    # 10% of training data is reversed for label
                    label_length = int(0.1 * len(self.data))
                    # random_idx = np.random.randint(len(self.train_data), size=label_length)
                    index = np.arange(len(self.data))
                    shuffle(index)
                    label_index = index[0: label_length]
                    unlabel_index = index[label_length:]
                    # Separate label data
                    label_data = self.data[label_index, :]
                    # print(type(label_data))
                    label_labels = itemgetter(*label_index)(self.labels)
                    with open(os.path.join(self.root, 'label.pt'), 'wb') as f:
                        torch.save((label_data, label_labels), f)
                    print('Save label data: %s, label labels: %d' % (label_data.shape, len(label_labels)))
                    # Separate unlabel data
                    unlabel_data = self.data[unlabel_index, :]
                    unlabel_labels = itemgetter(*unlabel_index)(self.labels)
                    with open(os.path.join(self.root, 'unlabel.pt'), 'wb') as f:
                        torch.save((unlabel_data, unlabel_labels), f)
                    print('Save unlabel data: %s, unlabel labels: %d' % (unlabel_data.shape, len(unlabel_labels)))
                    self.data, self.labels = torch.load(
                        os.path.join(self.root, '%s.pt' % add_split))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
