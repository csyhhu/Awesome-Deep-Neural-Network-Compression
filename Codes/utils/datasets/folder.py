import torch.utils.data as data

from PIL import Image
import os
import os.path
try:
    import lmdb
    import caffe
except:
    pass

import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        # with Image.open(f) as img:
        #     return img.convert('RGB')
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, data_path=None, transform=None, target_transform=None,
                 loader=default_loader, Train=True):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.Train = Train
        self.lmdb_dir = data_path
        if self.Train:
            self.lmdb_dir = self.lmdb_dir+'/ilsvrc12_train_lmdb/'
        else:
            self.lmdb_dir = self.lmdb_dir+'/ilsvrc12_val_lmdb/'
        self.lmdb_env = lmdb.open(self.lmdb_dir, readonly=True)
        self.lmdb_txn = self.lmdb_env.begin()

        self.length = self.lmdb_env.stat()['entries']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        datum = caffe.proto.caffe_pb2.Datum()
        lmdb_cursor = self.lmdb_txn.cursor()
        key_index ='{:08}'.format(index)
        value = lmdb_cursor.get(key_index.encode())
        datum.ParseFromString(value)
        # data = (caffe.io.datum_to_array(datum))
        # print('Min: %.3f, Max: %.3f' %(np.min(data), np.max(data)))
        # input()
        data = (caffe.io.datum_to_array(datum)) / (255.)
        # print(data.shape)
        if self.transform is not None:
            img = self.transform(data)
        target = datum.label

        return img, target

    def __len__(self):
        return self.length
