import os
import random
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.transforms import RandomHorizontalFlip
from torch.utils.data import Dataset
from torchvision import transforms as T


class MVTEC(Dataset):
    
    CLASS_NAMES = ['weld']
    
    def __init__(self, 
                 root: str,
                 class_name: str = 'bottle', 
                 train: bool = True,
                 normalize: str = 'imagebind',
                 **kwargs) -> None:
    
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('msk_crp_size'), kwargs.get('msk_crp_size')]
        
        if isinstance(self.class_name, str):
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_data(self.class_name)
        elif self.class_name is None:
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_all_data()
        else:
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_all_data(self.class_name)
        
        # set transforms
        if normalize == "imagenet":
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size', 224), T.InterpolationMode.BICUBIC),
                T.CenterCrop(kwargs.get('crp_size', 224)),
                T.ToTensor(),
                T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])])
        else:
            self.transform = T.Compose( 
                [
                    T.Resize(
                        kwargs.get('img_size', 224), interpolation=T.InterpolationMode.BICUBIC
                    ),
                    T.CenterCrop(kwargs.get('crp_size', 224)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        
        # mask
        self.target_transform = T.Compose([
            T.Resize(kwargs.get('msk_size', 224), T.InterpolationMode.NEAREST),
            T.CenterCrop(kwargs.get('msk_crp_size', 224)),
            T.ToTensor()])
        
        self.class_to_idx = {'weld': 0}
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, label, mask_path, class_name = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.class_names[idx]
        img, label, mask = self._load_image_and_mask(image_path, label, mask_path)
        
        return img, label, mask, class_name
    
    def _load_image_and_mask(self, image_path, label, mask_path):
        img = Image.open(image_path).convert('RGB')
        class_name = image_path.split('/')[-4]
    
        img = self.transform(img)
        #
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask_path)
            mask = self.target_transform(mask)
        
        return img, label, mask

    def _load_data(self, class_name):
        image_paths, labels, mask_paths = [], [], []
        phase = 'train' if self.train else 'test'
        
        image_dir = os.path.join(self.root, class_name, phase)
        mask_dir = os.path.join(self.root, class_name, 'ground_truth')

        img_types = sorted(os.listdir(image_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(image_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)
                                    if f.endswith(('.png', '.jpg', '.jpeg'))])
            image_paths.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                labels.extend([0] * len(img_fpath_list))
                mask_paths.extend([None] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(mask_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)
                    
        class_names = [class_name] * len(image_paths)
        return image_paths, labels, mask_paths, class_names
    
    def _load_all_data(self, class_names=None):
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_class_names = []
        CLASS_NAMES = class_names if class_names is not None else self.CLASS_NAMES
        for class_name in CLASS_NAMES:
            image_paths, labels, mask_paths, class_names = self._load_data(class_name)
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)
            all_mask_paths.extend(mask_paths)
            all_class_names.extend(class_names)
        return all_image_paths, all_labels, all_mask_paths, all_class_names


def get_normal_image_paths_mvtec(root, class_name):
    phase = 'train' 
    image_paths = []

    image_dir = os.path.join(root, class_name, phase)
    
    img_types = sorted(os.listdir(image_dir))
    for img_type in img_types:
        # load images
        img_type_dir = os.path.join(image_dir, img_type)
        if not os.path.isdir(img_type_dir):
            continue
        img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)])
        image_paths.extend(img_fpath_list)

    return image_paths