import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets.mvtec import MVTEC
from datasets.visa import VISA
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD
from models.imagebind import ImageBindModel
from models.clip import ClipModel
from models.dino import DinoModel
from models.projector import MultiScaleAttentionProjector
from ad_models.glass.train_val import train_val


def load_mc_reference_features(root_dir: str, class_names, device: torch.device, num_shot=4):
    refs = {}
    for class_name in class_names:
        layer1_refs = np.load(os.path.join(root_dir, class_name, 'layer1.npy'))
        layer2_refs = np.load(os.path.join(root_dir, class_name, 'layer2.npy'))
        layer3_refs = np.load(os.path.join(root_dir, class_name, 'layer3.npy'))
        layer4_refs = np.load(os.path.join(root_dir, class_name, 'layer4.npy'))
        
        layer1_refs = torch.from_numpy(layer1_refs).to(device)
        layer2_refs = torch.from_numpy(layer2_refs).to(device)
        layer3_refs = torch.from_numpy(layer3_refs).to(device)
        layer4_refs = torch.from_numpy(layer4_refs).to(device)
        
        K1 = (layer1_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer1_refs = layer1_refs[:K1, :]
        K2 = (layer2_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer2_refs = layer2_refs[:K2, :]
        K3 = (layer3_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer3_refs = layer3_refs[:K3, :]
        K4 = (layer4_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer4_refs = layer4_refs[:K4, :]
        
        refs[class_name] = (layer1_refs, layer2_refs, layer3_refs, layer4_refs)
    
    return refs


def get_dataset(
        name='mvtec',
        data_path="./data/mvtec_anomaly_detection",
        aug_path="./data/dtd/images",
        subdatasets=('carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'),
        batch_size=8,
        resize=224,
        imagesize=224,
        num_workers=16,
        rotate_degrees=0,
        translate=0,
        scale=0.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        gray=0.0,
        hflip=0.0,
        vflip=0.0,
        distribution=3,  # hypersphere
        mean=0.5,
        std=0.1,
        fg=1,  # foreground mask
        rand_aug=1,
        augment=False,
        normalize='imagenet',
        encoder_name='imagebind'
):
    fg = 1 if name == 'mvtec' or name == 'visa' else 0
    _DATASETS = {"mvtec": ["ad_models.glass.datasets.mvtec", "MVTecDataset"], "visa": ["ad_models.glass.datasets.visa", "VisADataset"],
                 "btad": ["ad_models.glass.datasets.btad", "BTADDataset"], "mvtec3d": ["ad_models.glass.datasets.mvtec3d", "MVTec3DDataset"],
                 "mpdd": ["ad_models.glass.datasets.mpdd", "MPDDDataset"]}
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    dataloaders = []
    for subdataset in subdatasets:
        test_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            aug_path,
            classname=subdataset,
            resize=resize,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TEST,
            seed=0,
            normalize=normalize,
            encoder_name=encoder_name
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
        )

        test_dataloader.name = name + "_" + subdataset

        test = 'ckpt'
        if test == 'ckpt':
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                aug_path,
                dataset_name=name,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=0,
                rotate_degrees=rotate_degrees,
                translate=translate,
                brightness_factor=brightness,
                contrast_factor=contrast,
                saturation_factor=saturation,
                gray_p=gray,
                h_flip_p=hflip,
                v_flip_p=vflip,
                scale=scale,
                distribution=distribution,
                mean=mean,
                std=std,
                fg=fg,
                rand_aug=rand_aug,
                augment=augment,
                batch_size=batch_size,
                normalize=normalize,
                encoder_name=encoder_name
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            train_dataloader.name = test_dataloader.name
        else:
            train_dataloader = test_dataloader

        dataloader_dict = {
            "training": train_dataloader,
            "testing": test_dataloader,
        }
        dataloaders.append(dataloader_dict)

    return dataloaders


def main(args):   
    if args.backbone == 'imagebind':
        encoder = ImageBindModel(args.backbone, device=args.device)
        projectors = MultiScaleAttentionProjector(encoder.feature_dimensions, device=args.device)
        normalize = 'imagebind'
    elif 'clip' in args.backbone:
        encoder = ClipModel(args.backbone, device=args.device)
        projectors = MultiScaleAttentionProjector(encoder.feature_dimensions, device=args.device)
        normalize = 'imagebind'
    elif 'dino' in args.backbone:
        encoder = DinoModel(args.backbone, device=args.device)
        projectors = MultiScaleAttentionProjector(encoder.feature_dimensions, device=args.device)
        normalize = 'imagenet'
    else:
        raise ValueError("Unrecognized Backbone!") 
    encoder = encoder.to(args.device)
    
    if args.with_pretrained:
        checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
        state_dict = checkpoint['projectors']
        projectors.load_state_dict(state_dict, strict=True)
        projectors = projectors.to(args.device)
        projectors.eval()
    else:
        projectors = None
    
    if args.dataset == 'mvtec':
        CLASS_NAMES = MVTEC.CLASS_NAMES
    elif args.dataset == 'visa':
        CLASS_NAMES = VISA.CLASS_NAMES
    elif args.dataset == 'btad':
        CLASS_NAMES = BTAD.CLASS_NAMES
    elif args.dataset == 'mvtec3d':
        CLASS_NAMES = MVTEC3D.CLASS_NAMES
    elif args.dataset == 'mpdd':
        CLASS_NAMES = MPDD.CLASS_NAMES
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset}!")
    
    dataloaders = get_dataset(name=args.dataset, data_path=args.dataset_dir, subdatasets=CLASS_NAMES, normalize=normalize, encoder_name=args.backbone)
    if args.with_pretrained:
        ref_features = load_mc_reference_features(args.ref_feature_dir, CLASS_NAMES, args.device, num_shot=args.num_ref_shot)
    else:
        ref_features = None
    img_aucs, img_aps, pix_aucs, pix_aps, pix_aupros = train_val(encoder, projectors, dataloaders, ref_features, CLASS_NAMES, args.device, args.with_pretrained)
                
    for idx, class_name in enumerate(CLASS_NAMES):
        print("Class Name: {}, Image AUC | AP: {} | {}, Pixel AUC | AP | AUPRO: {} | {} | {}".format(
                class_name, img_aucs[idx], img_aps[idx], pix_aucs[idx], pix_aps[idx], pix_aupros[idx]))
    print('Average Image AUC | AP: {:.3f} | {:.3f}, Average Pixel AUC | AP | AUPRO: {:.3f} | {:.3f} | {:.3f}'.format(
        np.mean(img_aucs), np.mean(img_aps), np.mean(pix_aucs), np.mean(pix_aps), np.mean(pix_aupros)))


if __name__ == '__main__':
    TOTAL_SHOT = 8
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default="dinov2-large")
    parser.add_argument('--with_pretrained', default=True, action="store_true")
    parser.add_argument('--pretrained_weights', type=str, default="./checkpoints/dinov2-large/checkpoints_pro_norm.pth")
    
    parser.add_argument('--dataset', type=str, default="mvtec")
    parser.add_argument('--dataset_dir', type=str, default="./data/mvtec_anomaly_detection")
    parser.add_argument('--ref_feature_dir', type=str, default="./ref_features/dinov2-large/mvtec_8shot")
    
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--num_ref_shot", type=int, default=8)
    
    args = parser.parse_args()
    
    main(args)
