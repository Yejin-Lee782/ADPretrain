import os
import warnings
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets.weld import WeldAD
# from datasets.real_iad import RealIAD
from models.imagebind import ImageBindModel
from models.clip import ClipModel
from models.dino import DinoModel
from utils import init_seeds, get_residual_features, get_mc_matched_ref_features

warnings.filterwarnings('ignore')
NORMALIZE = 'imagenet'

def main(args):
    global NORMALIZE
    if args.backbone == 'imagebind':
        NORMALIZE = 'imagebind'
    elif 'clip' in args.backbone:
        NORMALIZE = 'imagebind'
    elif 'dino' in args.backbone:
        NORMALIZE = 'imagenet'
    else:
        NORMALIZE = 'imagenet'
    train_dataset = WeldAD(args.data_path, class_name=None, train=True, 
                            normalize=NORMALIZE,
                            with_aug=False,
                            img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=8, drop_last=True
    )
    if args.backbone == 'imagebind':
        encoder = ImageBindModel(args.backbone, device=args.device)
    elif 'clip' in args.backbone:
        encoder = ClipModel(args.backbone, device=args.device)
    elif 'dino' in args.backbone:
        encoder = DinoModel(args.backbone, device=args.device)
    else:
        raise ValueError("Unrecognized Backbone!")
    encoder.to(args.device)
    get_normal_rfeatures_center(encoder, train_dataset, train_loader, args.save_path)
    

def get_normal_rfeatures_center(encoder, train_dataset, train_loader, save_path):  
    centers = [None, None, None, None]
    nums = [0, 0, 0, 0]
    progress_bar = tqdm(total=len(train_loader))
    progress_bar.set_description(f"Extract Features")
    for step, batch in enumerate(train_loader):
        images, labels, masks, class_names, views = batch
        
        if torch.sum(labels == 0) == 0:  # no normal
            continue
        images = images[labels == 0]
        images = images.to(args.device)
        class_names_ = []
        for class_name, label in zip(class_names, labels.tolist()):
            if label == 0:
                class_names_.append(class_name)
        views_ = []
        for view, label in zip(views, labels.tolist()):
            if label == 0:
                views_.append(view)
        
        with torch.no_grad():
            features = encoder.encode_image_from_tensors(images, shape='img')
        
        ref_features = get_mc_reference_features(encoder, train_dataset, class_names_, views_, images.device, 4)
        mfeatures = get_mc_matched_ref_features(features, class_names_, ref_features)
        rfeatures = get_residual_features(features, mfeatures)
        
        for l in range(len(rfeatures)): 
            e = rfeatures[l]  
            bs, dim, h, w = e.size()
            e = e.permute(0, 2, 3, 1).reshape(-1, dim)
            
            if centers[l] is None:
                centers[l] = torch.mean(e, dim=0)
                nums[l] = e.shape[0]
            else:
                centers[l] = (centers[l] * nums[l] + torch.sum(e, dim=0)) / (nums[l] + e.shape[0])
                nums[l] = nums[l] + e.shape[0]
        progress_bar.update(1)
    progress_bar.close()
    centers = torch.stack(centers, dim=0)
    centers = centers.cpu().numpy()
    np.save(save_path, centers)
    
    return centers
    
    
def get_mc_reference_features(encoder, dataset, class_names, views, device, num_shot=4):
    reference_features = {}
    # class_names = np.unique(class_names)
    for idx, (class_name, view) in enumerate(zip(class_names, views)):
        normal_paths = dataset.get_random_normal_images(class_name, view, num_shot)
        images = load_and_transform_vision_data(normal_paths, device)
        with torch.no_grad():
            features = encoder.encode_image_from_tensors(images.to(device), shape='seq')
            for l in range(len(features)):
                _, _, c = features[l].shape
                features[l] = features[l].reshape(-1, c)
            reference_features[idx] = features
    return reference_features


def load_and_transform_vision_data(image_paths, device):
    if NORMALIZE == 'imagebind':
        mean=(0.48145466, 0.4578275, 0.40821073)
        std=(0.26862954, 0.26130258, 0.27577711)
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        data_transform = T.Compose([
                T.Resize(224, T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)])
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)
                    
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--backbone', type=str, default="dinov2-large")
    parser.add_argument('--data_path', type=str, default="./data/Real-IAD")
    parser.add_argument('--save_path', type=str, default="centers/dino_large_rfeature_centers.npy")
    
    args = parser.parse_args()
    init_seeds(42)
    
    # enable tf32 for accelating
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    main(args)
    

    
    
            