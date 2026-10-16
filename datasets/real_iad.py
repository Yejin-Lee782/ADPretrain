import os
import cv2
import json
import random
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
    
class MoCoTransform:
    def __init__(self, img_size=224):
        self.transform = T.Compose([
            # T.RandomResizedCrop(224, scale=(0.2, 1.)),  can try
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            # T.RandomHorizontalFlip(), can try
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711))])
        
    def __call__(self, x):
        x_1 = self.transform(x)
        x_2 = self.transform(x)
        
        return x_1, x_2
    
    
class MoCoTransformWithMask:
    def __init__(self):
        self.transform = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.2, 1.), interpolation=cv2.INTER_CUBIC, p=0.5), 
            A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
            A.CenterCrop(224, 224),
            A.OneOf([
                A.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            A.ToGray(p=0.2),
            A.OneOf([A.GaussianBlur(sigma_limit=[.1, 2.])], p=0.5),
            A.HorizontalFlip()])
        
        self.t_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        self.m_transform = T.Compose([
            T.ToTensor()])

    def __call__(self, x, mask):
        transformed1 = self.transform(image=x, mask=mask)
        transformed2 = self.transform(image=x, mask=mask)
        x_1 = transformed1['image']
        x_2 = transformed2['image']
        m_1 = transformed1['mask']
        m_2 = transformed2['mask']
        x_1 = self.t_transform(x_1)
        x_2 = self.t_transform(x_2)
        m_1 = self.m_transform(m_1)
        m_2 = self.m_transform(m_2)
        
        return (x_1, x_2), (m_1, m_2)
    
    
class RealIAD(Dataset):
    
    CLASS_NAMES = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser',
                   'fire_hood', 'mint', 'mounts', 'pcb', 'phone_battery',
                   'plastic_nut', 'plastic_plug', 'porcelain_doll', 'regulator', 'rolled_strip_base',
                   'sim_card_set', 'switch', 'tape', 'terminalblock', 'toothbrush',
                   'toy', 'toy_brick', 'transistor1', 'u_block', 'usb',
                   'usb_adaptor', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    
    def __init__(
            self, 
            root: str,
            class_name: str,
            train: bool = True,
            normalize: str = 'imagebind',
            with_aug: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            **kwargs):
        
        self.sub_dir = 'realiad_512'
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('msk_crp_size', 224), kwargs.get('msk_crp_size', 224)]
        
        # load dataset
        if isinstance(self.class_name, str):
            self.data_items_by_class, self.data_items_by_view = self._load_data(self.class_name)
        elif self.class_name is None:
            self.data_items_by_class, self.data_items_by_view = self._load_all_data()
        else:
            self.data_items_by_class, self.data_items_by_view = self._load_all_data(self.class_name)
        
        self.normal_images = self._get_normal_images()
        self.num_images_per_view = self._get_num_images_per_view(self.data_items_by_view)
            
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
        
        if with_aug:
            self.transform = MoCoTransform(kwargs.get('crp_size', 224))
            # self.transform = MoCoTransformWithMask()
        
        # mask
        self.target_transform = T.Compose([
            T.Resize(kwargs.get('msk_size', 224), T.InterpolationMode.NEAREST),
            T.CenterCrop(kwargs.get('msk_crp_size', 224)),
            T.ToTensor()])
        
        self.class_to_idx = {'audiojack': 0, 'bottle_cap': 1, 'button_battery': 2, 'end_cap': 3, 'eraser': 4,
                             'fire_hood': 5, 'mint': 6, 'mounts': 7, 'pcb': 8, 'phone_battery': 9,
                             'plastic_nut': 10, 'plastic_plug': 11, 'porcelain_doll': 12, 'regulator': 13, 'rolled_strip_base': 14,
                             'sim_card_set': 15, 'switch': 16, 'tape': 17, 'terminalblock': 18, 'toothbrush': 19,
                             'toy': 20, 'toy_brick': 21, 'transistor1': 22, 'u_block': 23, 'usb': 24,
                             'usb_adaptor': 25, 'vcpill': 26, 'wooden_beads': 27, 'woodstick': 28, 'zipper': 29}
        self.idx_to_class = {0: 'audiojack', 1: 'bottle_cap', 2: 'button_battery', 3: 'end_cap', 4: 'eraser',
                             5: 'fire_hood', 6: 'mint', 7: 'mounts', 8: 'pcb', 9: 'phone_battery',
                             10: 'plastic_nut', 11: 'plastic_plug', 12: 'porcelain_doll', 13: 'regulator', 14: 'rolled_strip_base',
                             15: 'sim_card_set', 16: 'switch', 17: 'tape', 18: 'terminalblock', 19: 'toothbrush',
                             20: 'toy', 21: 'toy_brick', 22: 'transistor1', 23: 'u_block', 24: 'usb',
                             25: 'usb_adaptor', 26: 'vcpill', 27: 'wooden_beads', 28: 'woodstick', 29: 'zipper'}
    
    def __len__(self):
        if isinstance(self.data_items_by_view, dict):  # only one class
            num_images = self._get_num_images(self.data_items_by_view)
        elif isinstance(self.data_items_by_view, list):
            num_images = 0
            for idx in range(len(self.data_items_by_view)):  # for each class
                num_images += self._get_num_images(self.data_items_by_view[idx])
        else:
            raise ValueError("The data_items must be dict or list!")
        
        return num_images
    
    def __getitem__(self, idx):
        image_view, idx_ = self._convert_idx(idx)
        
        data_item = self.data_items_by_view[image_view][idx_]
        image_path = data_item["image_path"]
        label = data_item["label"]
        mask_path = data_item["mask_path"]
        class_name = data_item["class_name"]
        
        image, mask = self._load_image_and_mask(image_path, label, mask_path)
        # image, mask = self._load_image_and_mask2(image_path, label, mask_path)
        
        return image, label, mask, class_name, image_view
    
    def _get_normal_images(self):
        normal_images = {}
        for class_name in self.CLASS_NAMES:
            normal_images_by_view = {'view1': [], 'view2': [], 'view3': [], 'view4': [], 'view5': []}
            normal_image_dir = os.path.join(self.root, self.sub_dir, class_name, 'OK')
            samples = sorted(os.listdir(normal_image_dir))
            for sample in samples:
                sample_path = os.path.join(normal_image_dir, sample)
                image_files = sorted(os.listdir(sample_path))
                image_files = [image_file for image_file in image_files if image_file.endswith('.jpg')]
                for image_file in image_files:
                    image_view = self._get_image_view(image_file)
                    abs_image_path = os.path.join(sample_path, image_file)
                    normal_images_by_view[image_view].append(abs_image_path)
            normal_images[class_name] = normal_images_by_view
            
        return normal_images 
                    
    def get_random_normal_images(self, class_name, view, num_shot=4):         
        normal_images_by_class = self.normal_images[class_name]
        normal_images = normal_images_by_class[view]
        
        n_idxs = np.random.randint(len(normal_images), size=num_shot)
        n_idxs = n_idxs.tolist()
        normal_paths = []
        for n_idx in n_idxs:
            normal_paths.append(normal_images[n_idx])
        
        return normal_paths
        
    def _convert_idx(self, idx):
        num_images_cumsum = np.cumsum(list(self.num_images_per_view.values())).tolist()
        image_view, last_num = None, 0
        for view, cur_num in zip(self.num_images_per_view.keys(), num_images_cumsum):
            if idx < cur_num:
                image_view = view
                break
            else:
                last_num = cur_num
        idx = idx - last_num
        
        return image_view, idx
            
    def _load_image_and_mask(self, image_path, label, mask_path):
        image = Image.open(image_path).convert('RGB')
        
        image = self.transform(image)
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask[mask != 0] = 255
            mask = Image.fromarray(mask)
            mask = self.target_transform(mask)
        
        return image, mask
    
    def _load_image_and_mask2(self, image_path, label, mask_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        height, width, _= image.shape
        
        if label == 0:
            mask = np.zeros([height, width])
        else:
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask[mask != 0] = 255
            
        image, mask = self.transform(image, mask)
        
        return image, mask

    def _get_num_images(self, data_dict: dict):
        num_images = 0
        for key, val in data_dict.items():
            num_images += len(val)
        return num_images
    
    def _get_num_images_per_view(self, data_items: dict):
        num_images_per_view = {}
        for view, data in data_items.items():
            num_images_per_view[view] = len(data)
        return num_images_per_view

    def _get_image_view(self, image_file):
        if 'C1' in image_file:
            return 'view1'
        elif 'C2' in image_file:
            return 'view2'
        elif 'C3' in image_file:
            return 'view3'
        elif 'C4' in image_file:
            return 'view4'
        elif 'C5' in image_file:
            return 'view5'
        else:
            raise ValueError(f"The format of the image file {image_file} is not recognized!")
    
    def _get_image_label_and_mask(self, image_file, all_files):
        mask_file = image_file.split('.')[0] + '.png'
        if mask_file in all_files:
            return 1, mask_file
        else:
            return 0, None
        
    def _load_data(self, class_name):
        data_items = {'view1': [], 'view2': [], 'view3': [], 'view4': [], 'view5': []}
        
        normal_image_dir = os.path.join(self.root, self.sub_dir, class_name, 'OK')
        samples = sorted(os.listdir(normal_image_dir))
        for sample in samples:
            sample_path = os.path.join(normal_image_dir, sample)
            all_files = sorted(os.listdir(sample_path))
            image_files = [image_file for image_file in all_files if image_file.endswith('.jpg')]
            for image_file in image_files:
                image_view = self._get_image_view(image_file)
                abs_image_path = os.path.join(sample_path, image_file)
                label, mask_file = self._get_image_label_and_mask(image_file, all_files)
                if label == 1:
                    abs_mask_path = os.path.join(sample_path, mask_file)
                else:
                    abs_mask_path = None
                data_item = {"image_path": abs_image_path, "label": label,
                             "mask_path": abs_mask_path, "anomaly_type": "good",
                             "class_name": class_name}
                data_items[image_view].append(data_item)
        
        # can try filtered realiad, uncoment the following `if is_selected`
        # with open(f"./datasets/realiad/{class_name}.json", 'r') as f:
        #     filtered_samples = json.load(f)
            
        ng_image_dir = os.path.join(self.root, self.sub_dir, class_name, 'NG')
        anomaly_types = sorted(os.listdir(ng_image_dir))
        for anomaly_type in anomaly_types:
            samples = sorted(os.listdir(os.path.join(ng_image_dir, anomaly_type)))
            for sample in samples:
                sample_path = os.path.join(ng_image_dir, anomaly_type, sample)
                all_files = sorted(os.listdir(sample_path))
                image_files = [image_file for image_file in all_files if image_file.endswith('.jpg')]
                for image_file in image_files:    
                    # if is_selected(image_file, filtered_samples):
                    #     image_view = self._get_image_view(image_file)
                    #     abs_image_path = os.path.join(sample_path, image_file)
                    #     label, mask_file = self._get_image_label_and_mask(image_file, all_files)
                    #     if label == 1:
                    #         abs_mask_path = os.path.join(sample_path, mask_file)
                    #     else:
                    #         abs_mask_path = None
                    #     data_item = {"image_path": abs_image_path, "label": label,
                    #                 "mask_path": abs_mask_path, "anomaly_type": anomaly_type,
                    #                 "class_name": class_name}
                    #     data_items[image_view].append(data_item)
                    # else:
                    #     continue
                    
                    image_view = self._get_image_view(image_file)
                    abs_image_path = os.path.join(sample_path, image_file)
                    label, mask_file = self._get_image_label_and_mask(image_file, all_files)
                    if label == 1:
                        abs_mask_path = os.path.join(sample_path, mask_file)
                    else:
                        abs_mask_path = None
                    data_item = {"image_path": abs_image_path, "label": label,
                                "mask_path": abs_mask_path, "anomaly_type": anomaly_type,
                                "class_name": class_name}
                    data_items[image_view].append(data_item)
        
        return {class_name: data_items}, data_items
    
    def _load_all_data(self, class_names=None):
        all_data_items = {}
        CLASS_NAMES = class_names if class_names is not None else self.CLASS_NAMES
        for class_name in CLASS_NAMES:
            _, data_items = self._load_data(class_name)
            all_data_items[class_name] = data_items
        
        new_data_items = all_data_items[CLASS_NAMES[0]]
        for view in ['view1', 'view2', 'view3', 'view4', 'view5']:  # for all views
            for class_name in CLASS_NAMES[1:]:
                data = all_data_items[class_name]
                new_data_items[view].extend(data[view])
            
        return all_data_items, new_data_items


def is_selected(image_file, image_files):
    for _image_file in image_files:
        if image_file in _image_file:
            return True
    return False


if __name__ == '__main__':
    class_names = os.listdir('/data/data1/yxc/datasets/Real-IAD/realiad_512')
    class_names = sorted(class_names)
    print(class_names)
    class_to_idx = {}
    idx_to_class = {}
    for idx, class_name in enumerate(class_names):
        class_to_idx[class_name] = idx
        idx_to_class[idx] = class_name
    print(class_to_idx)
    print(idx_to_class)