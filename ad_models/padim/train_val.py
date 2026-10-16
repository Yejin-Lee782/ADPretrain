from typing import List
import random
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from torch import Tensor
import torch.nn.functional as F


def train_val(encoder, projectors, train_loader, test_loader, class_name, ref_features, device, with_pretrained):
    random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)

    train_outputs = OrderedDict([('layer0', []), ('layer1', []), ('layer2', []), ('layer3', [])])
    # extract train set features
    for batch in tqdm(train_loader, '| feature extraction | train | %s |' % class_name):
        images, _, _, _ = batch
        images = images.to(device)  # (b, 3, 224, 224)
        with torch.no_grad():
            features = encoder.encode_image_from_tensors(images)
            
            if with_pretrained:
                pfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, pfeatures)
                if projectors is not None:
                    features = projectors(*rfeatures)
                else:
                    features = rfeatures
            else:
                if projectors is not None:
                    features = projectors(*features)
                
        # get intermediate layer outputs
        for k, v in zip(train_outputs.keys(), features):
            train_outputs[k].append(v.cpu().detach())

    # every single feature level, calculate mean and cov statistics.
    for k, v in train_outputs.items():
        embedding_vectors = torch.cat(v, 0)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size() 
        embedding_vectors = embedding_vectors.view(B, C, H * W)

        # Calculate mean and cov in every position.
        # ====================================================================================
        # Using torch.mean and np.cov to calculate mean and cov
        mean = torch.mean(embedding_vectors, dim=0).numpy()  # (C, H*W)
        cov = np.zeros((C, C, H * W))
        I = np.identity(C)
        for i in range(H * W):
            # (B, C) -> (C, C)
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        cov = cov.astype(np.float32)
        # ====================================================================================

        # save learned distribution
        train_outputs[k] = [mean, cov]

    test_outputs = OrderedDict([('layer0', []), ('layer1', []), ('layer2', []), ('layer3', [])])
    gt_list, gt_mask_list = [], []
    # extract test set features
    for batch in tqdm(test_loader, '| feature extraction | test | %s |' % class_name):
        images, labels, masks, _ = batch
        gt_list.extend(labels.cpu().detach().numpy())
        gt_mask_list.extend(masks.cpu().detach().numpy())
        
        images = images.to(device)
        # model prediction
        with torch.no_grad():
            features = encoder.encode_image_from_tensors(images)
            
            if with_pretrained:
                pfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, pfeatures)
                if projectors is not None:
                    features = projectors(*rfeatures)
                else:
                    features = rfeatures
            else:
                if projectors is not None:
                    features = projectors(*features)
                
        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), features):
            test_outputs[k].append(v.cpu().detach())
    
    # every single feature level, calculate score_map
    score_maps_list = []
    for k, v in test_outputs.items():
        means = torch.from_numpy(train_outputs[k][0]).to(device).permute(1, 0).unsqueeze(1)  # (H*W, 1, C)
        covs = torch.from_numpy(train_outputs[k][1]).to(device).to(torch.float32).permute(2, 0, 1).unsqueeze(1)  # (H*W, 1, C, C)
        cov_invs = torch.inverse(covs)  # (H*W, 1, C, C)
        
        # if out of GPU memory
        # covs = torch.from_numpy(train_outputs[k][1]).to(torch.float32).permute(2, 0, 1).unsqueeze(1)  # (H*W, 1, C, C)
        # cov_invs = torch.inverse(covs).to(device)  # (H*W, 1, C, C)
        
        all_dist_list = []
        for v_i in v:
            embedding_vectors = v_i.to(device)
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)  

            dist_list = []
            for idx in range(embedding_vectors.shape[0]):
                samples = embedding_vectors[idx, :, :].transpose(0, 1)  # (H*W, C)
                dist_matrix = mahalanobis(samples, means, cov_invs)  # (H*W, )

                dist_list.append(dist_matrix)
            dist_list = torch.stack(dist_list, dim=0).reshape(B, H, W)  # (B, H, W)
            all_dist_list.append(dist_list)
            # =====================================================================================
        dist_list = torch.concatenate(all_dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=images.shape[2:], mode='bilinear',
                                  align_corners=True).squeeze()
        score_maps_list.append(score_map.squeeze())

    gt_list = np.asarray(gt_list)
    gt_mask = np.concatenate(gt_mask_list)
    
    score_maps = torch.mean(torch.stack(score_maps_list, -1), dim=-1).cpu().numpy()  # (N, 224, 224)
    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_maps[i] = gaussian_filter(score_maps[i], sigma=4)

    # Normalization
    max_score = score_maps.max()
    min_score = score_maps.min()
    scores = (score_maps - min_score) / (max_score - min_score)     
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)  
    
    return img_scores, scores, gt_list, gt_mask 


def get_matched_ref_features(features: List[Tensor], ref_features: List[Tensor]) -> List[Tensor]:
    """
    Get matched reference features for one class.
    """
    matched_ref_features = []
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
        feature_n = F.normalize(feature, p=2, dim=1)
        coreset = ref_features[layer_id]  # (N2, C)
        coreset_n = F.normalize(coreset, p=2, dim=1)
        
        dist = feature_n @ coreset_n.T
        cidx = torch.argmax(dist, dim=1)
        index_feats = coreset[cidx]
        index_feats = index_feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        matched_ref_features.append(index_feats)
    
    return matched_ref_features


def get_residual_features(features: List[Tensor], ref_features: List[Tensor], pos_flag: bool = False) -> List[Tensor]:
    residual_features = []
    for layer_id in range(len(features)):
        fi = features[layer_id]  # (B, dim, h, w)
        pi = ref_features[layer_id]  # (B, dim, h, w)
        
        if not pos_flag:
            ri = fi - pi
        else:
            ri = F.mse_loss(fi, pi, reduction='none')
        residual_features.append(ri)
    
    return residual_features


def mahalanobis(u, v, cov_inv):
    """
    Calculate mahalanobis distances among B1 feature vectors and its
    corresponding B2 feature vectors.
    Args:
        u(Tensor): shape (B1, C)
        v(Tensor): shape (B1, B2, C)
        cov_inv(Tensor): shape (B1, B2, C, C)
    Returns:
        m(Tensor): shape (B1, B2)
    """
    delta = (u.unsqueeze(1) - v).unsqueeze(2)  # (B1, B2, 1, C)
    x1 = torch.matmul(delta, cov_inv)  # (B1, B2, 1, C)
    m = torch.matmul(x1, delta.permute(0, 1, 3, 2)).squeeze()  # (B1, B2)
    return torch.sqrt(m)
