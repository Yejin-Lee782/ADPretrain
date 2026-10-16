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
from utils import calculate_metrics
from ad_models.padim.train_val import train_val


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
    
    if args.with_pretrained:
        ref_features = load_mc_reference_features(args.ref_feature_dir, CLASS_NAMES, args.device, num_shot=args.num_ref_shot)
    else:
        ref_features = None
    img_aucs, img_aps, img_f1_scores, pix_aucs, pix_aps, pix_f1_scores, pix_aupros = [], [], [], [], [], [], []
    for class_name in CLASS_NAMES:
        if args.dataset == 'mvtec':
            train_dataset = MVTEC(args.dataset_dir, class_name=class_name, train=True,
                                normalize=normalize,
                                img_size=256, crp_size=224, msk_size=256, msk_crp_size=224)
            test_dataset = MVTEC(args.dataset_dir, class_name=class_name, train=False,
                                normalize=normalize,
                                img_size=256, crp_size=224, msk_size=256, msk_crp_size=224)
        elif args.dataset == 'visa':
            train_dataset = VISA(args.dataset_dir, class_name=class_name, train=True,
                                normalize=normalize,
                                img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
            test_dataset = VISA(args.dataset_dir, class_name=class_name, train=False,
                                normalize=normalize,
                                img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        elif args.dataset == 'btad':
            train_dataset = BTAD(args.dataset_dir, class_name=class_name, train=True,
                                normalize=normalize,
                                img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
            test_dataset = BTAD(args.dataset_dir, class_name=class_name, train=False,
                                normalize=normalize,
                                img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        elif args.dataset == 'mvtec3d':
            train_dataset = MVTEC3D(args.dataset_dir, class_name=class_name, train=True,
                                normalize=normalize,
                                img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
            test_dataset = MVTEC3D(args.dataset_dir, class_name=class_name, train=False,
                                    normalize=normalize,
                                    img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        elif args.dataset == 'mpdd':
            train_dataset = MPDD(args.dataset_dir, class_name=class_name, train=True,
                                normalize=normalize,
                                img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
            test_dataset = MPDD(args.dataset_dir, class_name=class_name, train=False,
                                normalize=normalize,
                                img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        else:
            raise ValueError(f"Unrecognized dataset {args.dataset}!")
        
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
        
        img_scores, scores, labels, gt_masks = train_val(encoder, projectors, train_loader, test_loader, class_name, 
                                                         ref_features[class_name] if args.with_pretrained else None,
                                                         args.device, args.with_pretrained)
        img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(img_scores, scores, labels, gt_masks, pro=True, only_max_value=True)
        img_aucs.append(img_auc)
        img_aps.append(img_ap)
        img_f1_scores.append(img_f1_score)
        pix_aucs.append(pix_auc)
        pix_aps.append(pix_ap)
        pix_f1_scores.append(pix_f1_score)
        pix_aupros.append(pix_aupro)
                
    for idx, class_name in enumerate(CLASS_NAMES):
        print("Class Name: {}, Image AUC | AP | F1_Score: {} | {} | {}, Pixel AUC | AP | F1_Score | AUPRO: {} | {} | {} | {}".format(
                class_name, img_aucs[idx], img_aps[idx], img_f1_scores[idx], pix_aucs[idx], pix_aps[idx], pix_f1_scores[idx], pix_aupros[idx]))
    print('Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
        np.mean(img_aucs), np.mean(img_aps), np.mean(img_f1_scores), np.mean(pix_aucs), np.mean(pix_aps), np.mean(pix_f1_scores), np.mean(pix_aupros)))
    
             
if __name__ == '__main__':
    TOTAL_SHOT = 8
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default="dinov2-large")
    parser.add_argument('--with_pretrained', default=True, action="store_true")
    parser.add_argument('--pretrained_weights', type=str, default="./checkpoints/dinov2-large/checkpoints_pro_angle.pth")
    
    parser.add_argument('--dataset', type=str, default="mvtec")
    parser.add_argument('--dataset_dir', type=str, default="./data/mvtec_anomaly_detection")
    parser.add_argument('--ref_feature_dir', type=str, default="./ref_features/dinov2-large/mvtec_8shot")
    
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--num_ref_shot", type=int, default=8)
    
    args = parser.parse_args()
    
    main(args)
