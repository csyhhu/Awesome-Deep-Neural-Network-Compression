from __future__ import division
try:
    import caffe
except:
    pass
import torch
import math
import random
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections

import torchvision.transforms.functional as F


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.

    """

    def __call__(self, pic):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic)
            # backward compatibility
            return img.float()[[2,1,0]]

class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = channel - mean

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
    """

    # def __init__(self, mean=None, meanfile=None):
    #     if mean:
    #         self.mean = mean
    #     else:
    #         data = open(meanfile, 'rb').read()
    #         blob = caffe.proto.caffe_pb2.BlobProto()
    #         blob.ParseFromString(data)
    #         arr = np.array(caffe.io.blobproto_to_array(blob))
    #         self.mean = torch.from_numpy(arr[0].astype('float32'))
    #
    # def __call__(self, tensor):
    #     """
    #     Args:
    #         tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    #
    #     Returns:
    #         Tensor: Normalized image.
    #     """
    #     # TODO: make efficient
    #     for t, m in zip(tensor, self.mean):
    #         t.sub_(m)
    #     return tensor

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # print('Tensor shape in Normalize: %s' %tensor.shape)
        # print(tensor.shape)
        return F.normalize(tensor, self.mean, self.std)


class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        """
        assert(img.shape[1]==self.size)
        assert(img.shape[2]==self.size)
        return img

class CenterCrop(object):
    """Crops the given PIL.Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        w, h = (img.shape[1], img.shape[2])
        th, tw = self.size
        w_off = int((w - tw) / 2.)
        h_off = int((h - th) / 2.)
        img = img[:, h_off:h_off+th, w_off:w_off+tw]
        return img


class Pad(object):
    """Pad the given PIL.Image on all sides with the given "pad" value.

    Args:
        padding (int or sequence): Padding on each border. If a sequence of
            length 4, it is used to pad left, top, right and bottom borders respectively.
        fill: Pixel fill value. Default is 0.
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be padded.

        Returns:
            PIL.Image: Padded image.
        """
        return ImageOps.expand(img, border=self.padding, fill=self.fill)


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            img = np.flip(img, axis=2).copy()
        return img


class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        h_off = random.randint(0, img.shape[1]-self.size)
        w_off = random.randint(0, img.shape[2]-self.size)
        img = img[:, h_off:h_off+self.size, w_off:w_off+self.size]
        return img
