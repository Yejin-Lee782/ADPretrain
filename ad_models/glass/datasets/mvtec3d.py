from torchvision import transforms
from ..perlin import perlin_mask
from enum import Enum

import numpy as np
import pandas as pd

import PIL
from PIL import Image
import torch
import os
import glob

_CLASSNAMES = [
    'bagel',
    'cable_gland',
    'carrot',
    'cookie',
    'dowel',
    'foam',
    'peach',
    'potato',
    'rope',
    'tire',
]


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class MVTec3DDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
            self,
            source,
            anomaly_source_path='./datasets/dtd/images',
            dataset_name='mvtec3d',
            classname='bagel',
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            distribution=0,
            mean=0.5,
            std=0.1,
            fg=0,
            rand_aug=1,
            scale=0,
            batch_size=8,
            normalize='imagenet',
            encoder_name='imagebind',
            **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.batch_size = batch_size
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.fg = fg
        self.rand_aug = rand_aug
        self.resize = resize if self.distribution != 1 else [resize, resize]
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.classname = classname
        self.dataset_name = dataset_name
        self.encoder_name = encoder_name

        self.class_fg = 0  # without foreground mask

        self.image_paths, self.labels, self.mask_paths, self.img_types = self.get_image_data()
        self.anomaly_source_paths = sorted(1 * glob.glob(anomaly_source_path + "/*/*.jpg"))

        if normalize == 'imagenet':
            self.MEAN = [0.485, 0.456, 0.406]
            self.STD = [0.229, 0.224, 0.225]
        else:
            self.MEAN = (0.48145466, 0.4578275, 0.40821073)
            self.STD = (0.26862954, 0.26130258, 0.27577711)

        self.transform_img = [
            transforms.Resize(self.resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0 - scale, 1.0 + scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def rand_augmenter(self):
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_aug = [
            transforms.Resize(self.resize),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ]

        transform_aug = transforms.Compose(transform_aug)
        return transform_aug

    def __getitem__(self, idx):
        anomaly, image_path, mask_path = self.img_types[idx], self.image_paths[idx], self.mask_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        mask_fg = mask_s = aug_image = torch.tensor([1])
        if self.split == DatasetSplit.TRAIN:
            aug = Image.open(np.random.choice(self.anomaly_source_paths)).convert("RGB")
            if self.rand_aug:
                transform_aug = self.rand_augmenter()
                aug = transform_aug(aug)
            else:
                aug = self.transform_img(aug)

            mask_all = perlin_mask(image.shape, self.imgsize // self._get_stride(), 0, 6, mask_fg, 1)
            mask_s = torch.from_numpy(mask_all[0])
            mask_l = torch.from_numpy(mask_all[1])

            beta = np.random.normal(loc=self.mean, scale=self.std)
            beta = np.clip(beta, .2, .8)
            aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask_gt = PIL.Image.open(mask_path).convert('F')
            mask_gt = self.transform_mask(mask_gt)
            mask_gt = torch.where(mask_gt > 0, 1, 0).to(torch.float32)
        else:
            mask_gt = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "aug": aug_image,
            "mask_s": mask_s,
            "mask_gt": mask_gt,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.image_paths)

    def get_image_data(self):
        phase = 'train' if self.split == DatasetSplit.TRAIN else 'test'
        image_paths, labels, mask_paths, types = [], [], [], []

        img_dir = os.path.join(self.source, self.classname, phase)
        gt_dir = os.path.join(self.source, self.classname, 'test')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type, 'rgb')
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            image_paths.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                labels.extend([0] * len(img_fpath_list))
                mask_paths.extend([None] * len(img_fpath_list))
                types.extend(['good'] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type, 'gt')
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)
                types.extend([img_type] * len(img_fpath_list))

        return image_paths, labels, mask_paths, types

    def _get_stride(self):
        if self.encoder_name == 'clip-base':
            stride = 16
        elif self.encoder_name == 'dinov2-base':
            stride = 14
        elif self.encoder_name == 'clip-large' or self.encoder_name == 'dinov2-large':
            stride = 14
        elif self.encoder_name == 'imagebind':
            stride = 14
        else:
            raise ValueError("Unrecognized Encoder!") 
        
        return stride
