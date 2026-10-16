import os
import time
import yaml
import argparse
import logging
import numpy as np
from tqdm import tqdm
from easydict import EasyDict

import torch
import torch.optim
import torch.nn.functional as F
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
from utils import get_residual_features, load_and_transform_vision_data, get_matched_ref_features, calculate_metrics
from ad_models.uniad.models.model_helper import ModelHelper
from ad_models.uniad.utils.criterion_helper import build_criterion
from ad_models.uniad.utils.lr_helper import get_scheduler
from ad_models.uniad.utils.misc_helper import AverageMeter, set_random_seed
from ad_models.uniad.utils.optimizer_helper import get_optimizer


RANDOM_TRAIN_REF = False

def get_random_normal_images(class_name, num_shot=1):
    if class_name in MVTEC.CLASS_NAMES:
        path = os.path.join('/data/data1/yxc/datasets/mvtec_anomaly_detection', class_name, 'train', 'good')
    elif class_name in VISA.CLASS_NAMES:
        path = os.path.join('/data/data1/yxc/datasets/visa', class_name, 'Data', 'Images', 'Normal')
    else:
        raise ValueError('Unrecognized class_name!')
    
    filenames = os.listdir(path)
    n_idxs = np.random.randint(len(filenames), size=num_shot)
    n_idxs = n_idxs.tolist()
    normal_paths = []
    for n_idx in n_idxs:
        normal_paths.append(os.path.join(path, filenames[n_idx]))
    
    return normal_paths


def get_mc_reference_features(encoder, class_names, device):
    ref_features = {}
    class_names = np.unique(class_names)
    for class_name in class_names:
        normal_paths = get_random_normal_images(class_name)
        images = load_and_transform_vision_data(normal_paths, device, encoder.name)
        with torch.no_grad():
            features = encoder.encode_image_from_tensors(images, shape='seq')
            for i in range(len(features)):
                b, l, c = features[i].shape
                features[i] = features[i].reshape(-1, c)
            ref_features[class_name] = features
    return ref_features


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


def get_mc_matched_reference_features(features, class_names, ref_features):
    matched_ref_features = [[] for _ in range(len(features))]
    for idx, c in enumerate(class_names):  # for each image
        ref_features_c = ref_features[c]
        
        for layer_id in range(len(features)):  # for all layers of one image
            feature = features[layer_id][idx:idx+1]
            _, C, H, W = feature.shape
            
            feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
            feature_n = F.normalize(feature, p=2, dim=1)  # normalized features
            ref_feature = ref_features_c[layer_id]  # (N2, C)
            ref_feature_n = F.normalize(ref_feature, p=2, dim=1)  # normalized features
            
            dist = feature_n @ ref_feature_n.T  # (N1, N2)
            cidx = torch.argmax(dist, dim=1)
            index_feats = ref_feature[cidx]
            index_feats = index_feats.permute(1, 0).reshape(C, H, W)
            matched_ref_features[layer_id].append(index_feats)
            
    matched_ref_features = [torch.stack(item, dim=0) for item in matched_ref_features]
    
    return matched_ref_features


def create_logger(name, level=logging.INFO):
    log = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(sh)
    return log


def train_one_epoch(config, train_loader, encoder, projectors, model, ref_features, optimizer, lr_scheduler, epoch, start_iter, criterion, with_pretrained):
    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)

    model.train()

    logger = logging.getLogger("global_logger")
    end = time.time()
    for i, input in enumerate(train_loader):
        images, labels, masks, class_names = input
        images = images.to(config.device)
        input = {}
        input['image'] = images
        input['label'] = labels
        input['mask'] = masks
        input['clsname'] = class_names
        input['width'] = 224
        input['height'] = 224
        input['with_pretrained'] = with_pretrained
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        with torch.no_grad():
            features = encoder.encode_image_from_tensors(images)
        
            if with_pretrained:
                if RANDOM_TRAIN_REF:
                    ref_features = get_mc_reference_features(encoder, class_names, images.device)
                pfeatures = get_mc_matched_reference_features(features, class_names, ref_features)  # fixed ref-features, similar with random ref-features
                rfeatures = get_residual_features(features, pfeatures)
                rfeatures = projectors(*rfeatures)
                input['feature_align'] = torch.cat(rfeatures, dim=1)
            else:
                input['feature_align'] = torch.cat(features, dim=1)
                
        outputs = model(input)
        
        loss = 0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)
        losses.update(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.print_freq_step == 0:
            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=current_lr,
                )
            )

        end = time.time()


def validate(config, test_loader, encoder, projectors, model, ref_features, with_pretrained):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    with torch.no_grad():
        scores = []
        gt_list, gt_mask_list = [], []
        for i, input in tqdm(enumerate(test_loader)):
            image, label, mask, class_name = input
            gt_list.extend(label.cpu().numpy())
            gt_mask_list.append(mask.squeeze(1).cpu().numpy())
            input = {}
            image = image.to(config.device)
            input['image'] = image
            input['label'] = label
            input['mask'] = mask
            input['clsname'] = class_name
            input['width'] = 224
            input['height'] = 224
            # forward
            with torch.no_grad():
                features = encoder.encode_image_from_tensors(image)

            if with_pretrained:
                pfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, pfeatures)
                rfeatures = projectors(*rfeatures)
                input['feature_align'] = torch.cat(rfeatures, dim=1)
            else:
                input['feature_align'] = torch.cat(features, dim=1)
            
            outputs = model(input)
            preds = outputs["pred"].squeeze(1).cpu().numpy()
        
            scores.append(preds)

            # record loss
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["clsname"])
            losses.update(loss.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(test_loader), batch_time=batch_time
                    )
                )

    scores = np.concatenate(scores, axis=0)
    scores = np.array(scores, copy=True)
    gt_list = np.asarray(gt_list)
    gt_mask = np.concatenate(gt_mask_list, axis=0)
    gt_mask = np.array(gt_mask, copy=True)
    
    img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(None, scores, gt_list, gt_mask, pro=True, only_max_value=True)
    
    return img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro
    

def train(args):
    with open("./ad_models/uniad/config.yaml") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    config.device = args.device

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)
    create_logger("global_logger")
    
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
    
    # create model
    model = ModelHelper(config.net, encoder)
    model.to(args.device)
    
    active_layers = ['reconstruction']
    parameters = [
        {"params": getattr(model, layer).parameters()} for layer in active_layers
    ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)
    criterion = build_criterion(config.criterion)
    
    if args.dataset == 'mvtec':
        CLASS_NAMES = MVTEC.CLASS_NAMES
        train_dataset = MVTEC(args.dataset_dir, class_name=MVTEC.CLASS_NAMES, train=True,
                              normalize=normalize,
                              img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
    elif args.dataset == 'visa':
        CLASS_NAMES = VISA.CLASS_NAMES
        train_dataset = VISA(args.dataset_dir, class_name=VISA.CLASS_NAMES, train=True,
                              normalize=normalize,
                              img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
    elif args.dataset == 'btad':
        CLASS_NAMES = BTAD.CLASS_NAMES
        train_dataset = BTAD(args.dataset_dir, class_name=BTAD.CLASS_NAMES, train=True,
                              normalize=normalize,
                              img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
    elif args.dataset == 'mvtec3d':
        CLASS_NAMES = MVTEC3D.CLASS_NAMES
        train_dataset = MVTEC3D(args.dataset_dir, class_name=MVTEC3D.CLASS_NAMES, train=True,
                              normalize=normalize,
                              img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
    elif args.dataset == 'mpdd':
        CLASS_NAMES = MPDD.CLASS_NAMES
        train_dataset = MPDD(args.dataset_dir, class_name=MPDD.CLASS_NAMES, train=True,
                             normalize=normalize,
                             img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset}!")
    train_loader = DataLoader(
        train_dataset, batch_size=config.dataset['batch_size'], shuffle=True, num_workers=8, drop_last=True
    )

    if args.with_pretrained:
        ref_features = load_mc_reference_features(args.ref_feature_dir, CLASS_NAMES, args.device, num_shot=args.num_ref_shot)
    else:
        ref_features = None
    for epoch in range(config.trainer.max_epoch):
        last_iter = epoch * len(train_loader)
        train_one_epoch(config, train_loader, encoder, projectors, model, ref_features, optimizer,
                        lr_scheduler, epoch, last_iter, criterion, args.with_pretrained)
        lr_scheduler.step(epoch)
        
        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            img_aucs, img_aps, img_f1_scores, pix_aucs, pix_aps, pix_f1_scores, pix_aupros = [], [], [], [], [], [], []
            for class_name in CLASS_NAMES:
                if class_name in MVTEC.CLASS_NAMES:
                    test_dataset = MVTEC(args.dataset_dir, class_name=class_name, train=False,
                                        normalize=normalize,
                                        img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in VISA.CLASS_NAMES:
                    test_dataset = VISA(args.dataset_dir, class_name=class_name, train=False,
                                        normalize=normalize,
                                        img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in BTAD.CLASS_NAMES:
                    test_dataset = BTAD(args.dataset_dir, class_name=class_name, train=False,
                                        normalize=normalize,
                                        img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in MVTEC3D.CLASS_NAMES:
                    test_dataset = MVTEC3D(args.dataset_dir, class_name=class_name, train=False,
                                            normalize=normalize,
                                            img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in MPDD.CLASS_NAMES:
                    test_dataset = MPDD(args.dataset_dir, class_name=class_name, train=False,
                                        normalize=normalize,
                                        img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                else:
                    raise ValueError(f"Unrecognized dataset {args.dataset}!")
                test_loader = DataLoader(
                    test_dataset, batch_size=config.dataset['batch_size'], shuffle=False, num_workers=8, drop_last=False)
            
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = validate(config, test_loader, encoder, projectors, model,
                                                                                                   ref_features[class_name] if args.with_pretrained else None,
                                                                                                   args.with_pretrained)
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
    parser.add_argument('--pretrained_weights', type=str, default="./checkpoints/dinov2-large/checkpoints_pro_norm.pth")
    
    parser.add_argument('--dataset', type=str, default="mvtec")
    parser.add_argument('--dataset_dir', type=str, default="./data/mvtec_anomaly_detection")
    parser.add_argument('--ref_feature_dir', type=str, default="./ref_features/dinov2-large/mvtec_8shot")
    
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--num_ref_shot", type=int, default=8)
    
    args = parser.parse_args()
    
    train(args)