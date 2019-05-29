import os
import numpy as np
import random

from PIL import Image, ImageOps
from mxnet import ndarray as F
from mxnet import cpu
from gluoncv.data.segbase import SegmentationDataset
from mxnet.gluon.data.vision import transforms


class Fascades(SegmentationDataset):
    def __init__(self, root=os.path.expanduser('~/.mxnet/datasets/fascades'),
                 split='train', mode=None, transform=None, base_size=286, crop_size=256, **kwargs):
        super(Fascades, self).__init__(root, split, mode, transform, base_size, crop_size)
        self.images = _get_fascades(root, split)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        w, h = image.size
        img = image.crop(box=[0, 0, w // 2, h])
        mask = image.crop(box=[w // 2, 0, w, h])

        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        # todo: maybe changes to [256, 286] as org repo does
        short_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.BILINEAR)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _mask_transform(self, mask):
        return F.array(np.array(mask), cpu(0)).astype('float32')


def _transformer():
    """
    preprocess used in
        https://github.com/phillipi/pix2pix/blob/84db9dcfdf69cbfaf61ee5ece66a7802b8493f8e/data/donkey_folder.lua#L44
    :return:
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
    ])


def _get_fascades(folder, mode='train'):
    img_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'train')
    elif mode == 'val':
        img_folder = os.path.join(folder, 'val')
    else:
        img_folder = os.path.join(folder, 'test')

    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            img_paths.append(imgpath)
    return img_paths