import os
import random
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

from datasets.mvtec import MVTEC
from datasets.visa import VISA
from datasets.real_iad import RealIAD
from datasets.sampler import BalancedSampler

from validate import validate
from models.clip import ClipModel
from models.dino import DinoModel
from models.imagebind import ImageBindModel
from models.projector import MultiScaleAttentionProjector
from utils import init_seeds, get_residual_features, get_mc_matched_ref_features, patchify_mask
from losses.norm_loss import calculate_norm_loss
from losses.angle_loss import calculate_angle_loss
from classes import REALIAD_TO_MVTEC, REALIAD_TO_VISA

warnings.filterwarnings('ignore')

TOTAL_SHOT = 8  # total few-shot reference samples during testing
RANDOM_SHOT = True  # randomly change reference shot during training
WITH_AUG = True  # augment each image to two views
WITH_CENTER = True
DTYPE = torch.bfloat16
NORMALIZE = 'imagenet'


def main(args):
    CLASSES = REALIAD_TO_MVTEC
    global NORMALIZE
    if args.backbone == 'imagebind':
        NORMALIZE = 'imagebind'
    elif 'clip' in args.backbone:
        NORMALIZE = 'imagebind'
    elif 'dino' in args.backbone:
        NORMALIZE = 'imagenet'
    else:
        NORMALIZE = 'imagenet'
    
    train_dataset = RealIAD(args.train_dataset_dir, class_name=None, train=True, 
                            normalize=NORMALIZE,
                            with_aug=WITH_AUG,
                            img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    # balanced sampling
    # train_loader = DataLoader(
    #     train_dataset, batch_sampler=BalancedSampler(train_dataset, args.batch_size), num_workers=8
    # )

    if args.backbone == 'imagebind':
        encoder = ImageBindModel(args.backbone, device=args.device)
        projectors = MultiScaleAttentionProjector(encoder.feature_dimensions, device=args.device)
    elif 'clip' in args.backbone:
        encoder = ClipModel(args.backbone, device=args.device)
        projectors = MultiScaleAttentionProjector(encoder.feature_dimensions, device=args.device)
    elif 'dino' in args.backbone:
        encoder = DinoModel(args.backbone, device=args.device)
        projectors = MultiScaleAttentionProjector(encoder.feature_dimensions, device=args.device)
    else:
        raise ValueError("Unrecognized Backbone!")
    args.feature_levels = len(encoder.feature_dimensions)
    encoder.to(DTYPE)
    encoder.to(args.device)
    projectors.to(DTYPE)
    projectors.to(args.device)
    optimizer = torch.optim.Adam(projectors.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 9], gamma=0.1)
    
    best_pro, best_img_auc, best_pix_auc = 0, 0, 0
    best_pro_norm, best_img_auc_norm, best_pix_auc_norm = 0, 0, 0
    centers = np.load(args.feature_centers)
    centers = torch.from_numpy(centers).to(args.device)
    for epoch in range(args.epochs):
        projectors.train()
        
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}]")
        for step, batch in enumerate(train_loader):
            loss_angle_total, num1 = 0, 0
            loss_norm_total, num2 = 0, 0
            images, labels, masks, class_names, views = batch
            
            if torch.sum(labels) == 0:  # no anomaly, skip
                continue
            images1, images2 = images
            images1 = images1.to(dtype=DTYPE, device=args.device)
            images2 = images2.to(dtype=DTYPE, device=args.device)
            masks = masks.to(args.device)
            
            with torch.no_grad():
                features1 = encoder.encode_image_from_tensors(images1)
                features2 = encoder.encode_image_from_tensors(images2)
            
            ref_features = get_mc_reference_features(encoder, train_dataset, class_names, views, images1.device, args.train_ref_shot)
            mfeatures1 = get_mc_matched_ref_features(features1, class_names, ref_features)
            rfeatures1 = get_residual_features(features1, mfeatures1)
            mfeatures2 = get_mc_matched_ref_features(features2, class_names, ref_features)
            rfeatures2 = get_residual_features(features2, mfeatures2)
            
            lvl_masks1, lvl_masks2 = [], []
            for l in range(args.feature_levels):
                _, _, h, w = rfeatures1[l].size()
                m = patchify_mask(masks, patch_size=224 // h) 
                lvl_masks1.append(m)
                m = F.interpolate(masks, size=(h, w), mode='nearest').squeeze(1)
                lvl_masks2.append(m)
            
            rfeatures1 = projectors(*rfeatures1)
            rfeatures2 = projectors(*rfeatures2)
            loss = 0
            for l in range(args.feature_levels): 
                e1 = rfeatures1[l]  
                e2 = rfeatures2[l]
                bs, dim, h, w = e1.size()
                e1 = e1.permute(0, 2, 3, 1).reshape(-1, dim)
                e2 = e2.permute(0, 2, 3, 1).reshape(-1, dim)
                m1 = lvl_masks1[l]
                m1 = m1.reshape(-1)
                m2 = lvl_masks2[l]
                m2 = m2.reshape(-1)
                
                if WITH_CENTER:
                    center = centers[l]
                    loss_angle = calculate_angle_loss(e1, e2, m1, center)
                else:
                    loss_angle = calculate_angle_loss(e1, e2, m1)
                if torch.isnan(loss_angle) or torch.isinf(loss_angle):
                    loss_angle = 0
                    loss_angle_total += 0
                else:
                    loss_angle_total += loss_angle.item()
                num1 += 1
                    
                loss_norm1 = calculate_norm_loss(e1, m2)
                loss_norm2 = calculate_norm_loss(e2, m2)
                loss_norm = 0
                if torch.is_tensor(loss_norm1):
                    loss_norm += loss_norm1
                    loss_norm_total += loss_norm1.item()
                if torch.is_tensor(loss_norm2):
                    loss_norm += loss_norm2
                    loss_norm_total += loss_norm2.item()
                num2 += 1
                
                if isinstance(loss_angle, int) and isinstance(loss_norm, int): # some errors, loss_angle = 0 and loss_norm = 0
                    pass
                else:
                    loss_l = loss_angle + loss_norm
                loss += loss_l
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(angle_loss='{:.3f}'.format(loss_angle_total / num1),
                                     norm_loss='{:.3f}'.format(loss_norm_total / num2))
            progress_bar.update(1)
        scheduler.step()     
        progress_bar.close()
        
        if (epoch + 1) % args.eval_freq == 0:
            metrics_patchcore, metrics_norm = [], []
            ref_features = load_mc_reference_features(args.test_ref_feature_dir, CLASSES['unseen'], args.device, args.num_ref_shot)
            for class_name in CLASSES['unseen']:
                if class_name in MVTEC.CLASS_NAMES:
                    normal_dataset = MVTEC(args.test_dataset_dir, class_name=class_name, train=True,
                                         normalize=NORMALIZE,
                                         img_size=256, crp_size=224, msk_size=256, msk_crp_size=224)
                    test_dataset = MVTEC(args.test_dataset_dir, class_name=class_name, train=False,
                                         normalize=NORMALIZE,
                                         img_size=256, crp_size=224, msk_size=256, msk_crp_size=224)
                elif class_name in VISA.CLASS_NAMES:
                    normal_dataset = VISA(args.test_dataset_dir, class_name=class_name, train=True,
                                        normalize=NORMALIZE,
                                        img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                    test_dataset = VISA(args.test_dataset_dir, class_name=class_name, train=False,
                                        normalize=NORMALIZE,
                                        img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                else:
                    raise ValueError('Unrecognized class name: {}'.format(class_name))
                normal_loader = DataLoader(
                    normal_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False
                )
                metrics = validate(args, encoder, projectors, normal_loader, test_loader, ref_features[class_name],
                                   args.device, class_name, residual=True, pro=True, dtype=DTYPE)
                metrics_patchcore.append(metrics['metrics'])
                metrics_norm.append(metrics['metrics-norm'])
            
            for idx, class_name in enumerate(CLASSES['unseen']):
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = metrics_patchcore[idx]
                print("Epoch: {}, Class Name: {}, Image AUC | AP | F1_Score: {} | {} | {}, Pixel AUC | AP | F1_Score | AUPRO: {} | {} | {} | {}".format(
                    epoch, class_name, img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
            metrics_patchcore = np.array(metrics_patchcore)
            img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = np.mean(metrics_patchcore, axis=0)
            print('(PatchCore) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
            
            os.makedirs(args.checkpoint_path, exist_ok=True)
            if img_auc > best_img_auc:
                best_img_auc = img_auc
                state_dict = {'projectors': projectors.state_dict(), 'img_auc': img_auc, 'pix_auc': pix_auc, 'pix_pro': pix_aupro}
                torch.save(state_dict, os.path.join(args.checkpoint_path, f'checkpoints_img_angle.pth'))
            if pix_auc > best_pix_auc:
                best_pix_auc = pix_auc
                state_dict = {'projectors': projectors.state_dict(), 'img_auc': img_auc, 'pix_auc': pix_auc, 'pix_pro': pix_aupro}
                torch.save(state_dict, os.path.join(args.checkpoint_path, f'checkpoints_pix_angle.pth'))
            if pix_aupro > best_pro:
                best_pro = pix_aupro
                state_dict = {'projectors': projectors.state_dict(), 'img_auc': img_auc, 'pix_auc': pix_auc, 'pix_pro': pix_aupro}
                torch.save(state_dict, os.path.join(args.checkpoint_path, f'checkpoints_pro_angle.pth'))
            
            metrics_norm = np.array(metrics_norm)
            img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = np.mean(metrics_norm, axis=0)
            print('(Norm) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
            
            if img_auc > best_img_auc_norm:
                best_img_auc_norm = img_auc
                state_dict = {'projectors': projectors.state_dict(), 'img_auc': img_auc, 'pix_auc': pix_auc, 'pix_pro': pix_aupro}
                torch.save(state_dict, os.path.join(args.checkpoint_path, f'checkpoints_img_norm.pth'))
            if pix_auc > best_pix_auc_norm:
                best_pix_auc_norm = pix_auc
                state_dict = {'projectors': projectors.state_dict(), 'img_auc': img_auc, 'pix_auc': pix_auc, 'pix_pro': pix_aupro}
                torch.save(state_dict, os.path.join(args.checkpoint_path, f'checkpoints_pix_norm.pth'))
            if pix_aupro > best_pro_norm:
                best_pro_norm = pix_aupro
                state_dict = {'projectors': projectors.state_dict(), 'img_auc': img_auc, 'pix_auc': pix_auc, 'pix_pro': pix_aupro}
                torch.save(state_dict, os.path.join(args.checkpoint_path, f'checkpoints_pro_norm.pth'))
    
    
def load_mc_reference_features(root_dir: str, class_names, device: torch.device, num_shot=4):
    refs = {}
    for class_name in class_names:
        layer1_refs = np.load(os.path.join(root_dir, class_name, 'layer1.npy'))
        layer2_refs = np.load(os.path.join(root_dir, class_name, 'layer2.npy'))
        layer3_refs = np.load(os.path.join(root_dir, class_name, 'layer3.npy'))
        layer4_refs = np.load(os.path.join(root_dir, class_name, 'layer4.npy'))
        
        layer1_refs = torch.from_numpy(layer1_refs).to(dtype=DTYPE, device=device)
        layer2_refs = torch.from_numpy(layer2_refs).to(dtype=DTYPE, device=device)
        layer3_refs = torch.from_numpy(layer3_refs).to(dtype=DTYPE, device=device)
        layer4_refs = torch.from_numpy(layer4_refs).to(dtype=DTYPE, device=device)
        
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
    
    
def get_mc_reference_features(encoder, dataset, class_names, views, device, num_shot=4):
    if RANDOM_SHOT:
        num_shot = random.choice([1, 4, 8])  # every time, randomly change the reference shot
    
    reference_features = {}
    # class_names = np.unique(class_names)
    for idx, (class_name, view) in enumerate(zip(class_names, views)):
        # if RANDOM_SHOT:
        #     num_shot = random.choice([2, 4, 8])
        normal_paths = dataset.get_random_normal_images(class_name, view, num_shot)
        images = load_and_transform_vision_data(normal_paths, device)
        with torch.no_grad():
            features = encoder.encode_image_from_tensors(images.to(dtype=DTYPE, device=device), shape='seq')
            for l in range(len(features)):
                _, _, c = features[l].shape
                features[l] = features[l].reshape(-1, c)
            reference_features[idx] = features
    return reference_features
            
            
def load_and_transform_vision_data(image_paths, device):
    if NORMALIZE == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

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
    parser.add_argument('--train_dataset_dir', type=str, default="./data/Real-IAD")
    parser.add_argument('--test_dataset_dir', type=str, default="./data/mvtec_anomaly_detection")
    parser.add_argument('--test_ref_feature_dir', type=str, default="./ref_features/dinov2-large/mvtec_8shot")
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/dinov2-large")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="dinov2-large")
    parser.add_argument('--feature_centers', type=str, default="./centers/dino_large_rfeature_centers.npy")
    
    parser.add_argument('--feature_levels', default=4, type=int)
    parser.add_argument("--train_ref_shot", type=int, default=4)
    parser.add_argument("--num_ref_shot", type=int, default=8)
    
    args = parser.parse_args()
    init_seeds(42)
    
    # enable tf32 for accelating
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    main(args)
    

    
    
            