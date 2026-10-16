from sklearn import metrics
from skimage import measure

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
import torch


def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    """
    Computes the best precision, recall and threshold for a given set of
    anomaly ground truth labels and anomaly prediction weights.
    """
    precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_precision = precision[np.argmax(f1_scores)]
    best_recall = recall[np.argmax(f1_scores)]
    print(best_threshold, best_precision, best_recall)

    return best_threshold, best_precision, best_recall


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).
    """
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0. if path == 'training' else metrics.average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    return {"auroc": auroc, "ap": ap}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='train'):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    auroc = metrics.roc_auc_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)  # only == 1 will be 1, other <1 float will be 0
    ap = 0. if path == 'training' else metrics.average_precision_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)

    return {"auroc": auroc, "ap": ap}


def compute_pro(masks, amaps, num_th=200):
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, index=[0])])

    df = df[df["fpr"] < 0.3]
    df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc


def calculate_metrics(img_scores, scores, labels, gt_masks, pro=True, only_max_value=True):
    """
    Args:
        scores (np.ndarray): shape (N, H, W).
        labels (np.ndarray): shape (N, ), 0 for normal, 1 for abnormal.
        gt_masks (np.ndarray): shape (N, H, W).
    """
    # average precision
    pix_ap = round(average_precision_score(gt_masks.flatten(), scores.flatten()), 5)
    # f1 score, f1 score is to balance the precision and recall
    # f1 score is high means the precision and recall are both high
    precisions, recalls, _ = precision_recall_curve(gt_masks.flatten(), scores.flatten())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    pix_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
    # roc auc
    pix_auc = round(roc_auc_score(gt_masks.flatten(), scores.flatten()), 5)
    
    if img_scores is None:
        _, h, w = scores.shape
        size = h * w
        if only_max_value:
            topks = [1]
        else:
            topks = [int(size*p) for p in np.arange(0.01, 0.41, 0.01)]
            topks = [1, 100] + topks
        img_aps, img_aucs, img_f1_scores = [], [], []
        for topk in topks:
            img_scores = get_image_scores(scores, topk)
            img_ap = round(average_precision_score(labels, img_scores), 5)
            precisions, recalls, _ = precision_recall_curve(labels, img_scores)
            f1_scores = (2 * precisions * recalls) / (precisions + recalls)
            img_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
            img_auc = round(roc_auc_score(labels, img_scores), 5)
            img_aps.append(img_ap)
            img_aucs.append(img_auc)
            img_f1_scores.append(img_f1_score)
        img_ap, img_auc, img_f1_score = np.max(img_aps), np.max(img_aucs), np.max(img_f1_scores)
    else:
        img_ap = round(average_precision_score(labels, img_scores), 5)
        img_auc = round(roc_auc_score(labels, img_scores), 5)
        precisions, recalls, _ = precision_recall_curve(labels, img_scores)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        img_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
        
    if pro:
        pix_aupro = calculate_aupro(gt_masks, scores)
    else:
        pix_aupro = -1
    
    return img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro


def get_image_scores(scores, topk=1):
    scores_ = torch.from_numpy(scores)
    img_scores = torch.topk(scores_.reshape(scores_.shape[0], -1), topk, dim=1)[0]
    img_scores = torch.mean(img_scores, dim=1)
    img_scores = img_scores.cpu().numpy()
        
    return img_scores


def calculate_aupro(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    try:
        pros, fprs, ths = [], [], []
        for th in np.arange(min_th, max_th, delta):
            binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
            pro = []
            for binary_amap, mask in zip(binary_amaps, masks):
                for region in measure.regionprops(measure.label(mask)):
                    tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                    pro.append(tp_pixels / region.area)
            inverse_masks = 1 - masks
            fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
            fpr = fp_pixels / inverse_masks.sum()
            pros.append(np.array(pro).mean())
            fprs.append(fpr)
            ths.append(th)
        pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
        idxes = fprs < expect_fpr
        fprs = fprs[idxes]
        if fprs.shape[0] <= 2:
            return 0.5
        else:
            fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
            pro_auc = auc(fprs, pros[idxes])
            return pro_auc
    except:  
        return 0.5
