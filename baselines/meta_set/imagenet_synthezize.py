# -*- coding: utf-8 -*-

import os
from PIL import Image
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
from tqdm import trange
from PIL import Image
import random

# /////////////// Data Loader ///////////////


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


sometimes = lambda aug: iaa.Sometimes(0.5, aug)
list = [
    iaa.pillike.Autocontrast(),  # adjust contrast by cutting off p% of lowest/highest histogram values
    iaa.Multiply((0.1, 1.9), per_channel=0.2),  # make some images brighter and some darker
    iaa.pillike.EnhanceColor(),  # remove a random fraction of color from input images
    iaa.Solarize(0.5, threshold=(32, 128)),  # invert the colors
    iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-30, 30)
    )),  # add affine transformation
    iaa.Sharpen(alpha=(0.1, 1.0)),  # apply a sharpening filter kernel to images
]



def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


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
        img = Image.open(f)
        return img.convert('RGB')


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    return pil_loader(path)


class DistortImageFolder(data.Dataset):
    def __init__(self, root, method, number, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n" +
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.method = method
        self.number = number
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            img = np.asarray(img)
            img = self.method(image=img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        save_path = '/home/qhu/qhu-data/TS4code/meta_set/imagenet_data_3/' + str(self.number) + "/" +\
                    self.idx_to_class[target]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path += path[path.rindex('/'):]
        Image.fromarray(np.uint8(img)).save(save_path, quality=85, optimize=True)
        return 0  # we do not care about returning the data

    def __len__(self):
        return len(self.imgs)


# /////////////// Distortion Helpers ///////////////

import cv2
import warnings

warnings.simplefilter("ignore", UserWarning)


def auc(errs):  # area under the alteration error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
# /////////////// End Distortions ///////////////


# /////////////// Further Setup ///////////////


def save_distorted(method, number):
    # for severity in range(1, 6):
    distorted_dataset = DistortImageFolder(
        root="/home/qhu/qhu-data/Quan_study/datasets/tiny-imagenet/tiny-imagenet-200/val/",
        method=method, number=number,
        transform=trn.Compose([trn.Resize((64, 64))]))
    distorted_dataset_loader = torch.utils.data.DataLoader(
        distorted_dataset, batch_size=100, shuffle=False, num_workers=6)
    for _ in distorted_dataset_loader: continue


# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
import collections

print('\nUsing ImageNet data')

d = collections.OrderedDict()

num_sets = 1000
# for indice_set in trange(num_sets):
#     num_sel = 1
#     list_sel = random.sample(list, int(num_sel))
#     random.shuffle(list_sel)
#     seq = iaa.Sequential(list_sel)
#     save_distorted(seq, indice_set)

# for indice_set in trange(num_sets):
#     num_sel = 2
#     list_sel = random.sample(list, int(num_sel))
#     random.shuffle(list_sel)
#     seq = iaa.Sequential(list_sel)
#     save_distorted(seq, indice_set)

for indice_set in trange(num_sets):
    num_sel = 3
    list_sel = random.sample(list, int(num_sel))
    random.shuffle(list_sel)
    seq = iaa.Sequential(list_sel)
    save_distorted(seq, indice_set)
