import os
import math
import random
from typing import List, Dict
from PIL import Image
import numpy as np
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as T
from datasets.mvtec import MVTEC
from datasets.visa import VISA


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def patchify_mask(mask, patch_size=14):
    """
    Convert mask into patches.
    
    Args:
        imgs: shape of (N, 1, H, W).
    Returns:
        patches: (N, L, patch_size**2 * 1).
    """
    n, num_chans, _, _ = mask.shape
    assert mask.shape[2] == mask.shape[3] and mask.shape[2] % patch_size == 0

    h = w = mask.shape[2] // patch_size
    patches = mask.reshape(shape=(n, num_chans, h, patch_size, w, patch_size))
    patches = torch.einsum('nchpwq->nhwpqc', patches)
    patches = patches.reshape(shape=(n, h * w, patch_size**2 * num_chans))
    mask = patches.sum(dim=-1)
    # mask = (mask != 0).type(torch.float32)
    
    # threshold = patch_size * patch_size // 4
    threshold = 32
    mask = (mask > threshold).type(torch.float32)
    
    return mask

    
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


def get_random_normal_images(root, class_name, num_shot=4):
    if class_name in MVTEC.CLASS_NAMES:
        root_dir = os.path.join(root, class_name, 'train', 'good')
    elif class_name in VISA.CLASS_NAMES:
        root_dir = os.path.join(root, class_name, 'Data', 'Images', 'Normal')
    else:
        raise ValueError('Unrecognized class_name!')
    filenames = os.listdir(root_dir)
    n_idxs = np.random.randint(len(filenames), size=num_shot)
    n_idxs = n_idxs.tolist()
    normal_paths = []
    for n_idx in n_idxs:
        normal_paths.append(os.path.join(root_dir, filenames[n_idx]))
    
    return normal_paths


def get_mc_reference_features(encoder, root, class_names, device, num_shot=4):
    """
    Get reference features for multiple classes.
    """
    reference_features = {}
    class_names = np.unique(class_names)
    for class_name in class_names:
        normal_paths = get_random_normal_images(root, class_name, num_shot)
        images = load_and_transform_vision_data(normal_paths, device)
        with torch.no_grad():
            features = encoder(images)
            for l in range(len(features)):
                bs, c, h, w = features[l].shape
                features[l] = features[l].permute(0, 2, 3, 1).reshape(-1, c)
            reference_features[class_name] = features
    return reference_features


def load_and_transform_vision_data(image_paths, device, encoder_name):
    if encoder_name == 'imagebind':
        normalize = 'imagebind'
    elif 'clip' in encoder_name:
        normalize = 'imagebind'
    elif 'dino' in encoder_name:
        normalize = 'imagenet'
    else:
        raise ValueError("Unrecognized Backbone!") 
    
    if normalize == 'imagebind':
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


def get_mc_matched_ref_features(features: List[Tensor], class_names: List[str],
                                ref_features: Dict[int, List[Tensor]]) -> List[Tensor]:
    """
    Get matched reference features for multiple classes.
    """
    matched_ref_features = [[] for _ in range(len(features))]
    for idx, _ in enumerate(class_names):  # for each image
        ref_features_c = ref_features[idx]
        
        for layer_id in range(len(features)):  # for all layers of one image
            feature = features[layer_id][idx:idx+1]
            _, C, H, W = feature.shape
            
            feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
            feature_n = F.normalize(feature, p=2, dim=1)
            coreset = ref_features_c[layer_id]  # (N2, C)
            coreset_n = F.normalize(coreset, p=2, dim=1)
            
            dist = feature_n @ coreset_n.T  # (N1, N2)
            cidx = torch.argmax(dist, dim=1)
            index_feats = coreset[cidx]
            index_feats = index_feats.permute(1, 0).reshape(C, H, W)
            matched_ref_features[layer_id].append(index_feats)
            
    matched_ref_features = [torch.stack(item, dim=0) for item in matched_ref_features]
    
    return matched_ref_features


# def calculate_metrics(img_scores, scores, labels, gt_masks, pro=True, only_max_value=True):
#     """
#     Args:
#         scores (np.ndarray): shape (N, H, W).
#         labels (np.ndarray): shape (N, ), 0 for normal, 1 for abnormal.
#         gt_masks (np.ndarray): shape (N, H, W).
#     """
#     # average precision
#     pix_ap = round(average_precision_score(gt_masks.flatten(), scores.flatten()), 5)
#     # f1 score, f1 score is to balance the precision and recall
#     # f1 score is high means the precision and recall are both high
#     precisions, recalls, _ = precision_recall_curve(gt_masks.flatten(), scores.flatten())
#     f1_scores = (2 * precisions * recalls) / (precisions + recalls)
#     pix_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
#     # roc auc
#     pix_auc = round(roc_auc_score(gt_masks.flatten(), scores.flatten()), 5)
    
#     if img_scores is None:
#         _, h, w = scores.shape
#         size = h * w
#         if only_max_value:
#             topks = [1]
#         else:
#             topks = [int(size*p) for p in np.arange(0.01, 0.41, 0.01)]
#             topks = [1, 100] + topks
#         img_aps, img_aucs, img_f1_scores = [], [], []
#         for topk in topks:
#             img_scores = get_image_scores(scores, topk)
#             img_ap = round(average_precision_score(labels, img_scores), 5)
#             precisions, recalls, _ = precision_recall_curve(labels, img_scores)
#             f1_scores = (2 * precisions * recalls) / (precisions + recalls)
#             img_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
#             img_auc = round(roc_auc_score(labels, img_scores), 5)
#             img_aps.append(img_ap)
#             img_aucs.append(img_auc)
#             img_f1_scores.append(img_f1_score)
#         img_ap, img_auc, img_f1_score = np.max(img_aps), np.max(img_aucs), np.max(img_f1_scores)
#     else:
#         img_ap = round(average_precision_score(labels, img_scores), 5)
#         img_auc = round(roc_auc_score(labels, img_scores), 5)
#         precisions, recalls, _ = precision_recall_curve(labels, img_scores)
#         f1_scores = (2 * precisions * recalls) / (precisions + recalls)
#         img_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
        
#     if pro:
#         pix_aupro = calculate_aupro(gt_masks, scores)
#     else:
#         pix_aupro = -1
    
#     return img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro
def calculate_metrics(
    img_scores,
    scores,
    labels,
    gt_masks,
    pro=True,
    only_max_value=True,
):
    """
    Args:
        scores (np.ndarray): shape (N, H, W)
        labels (np.ndarray): shape (N,)
        gt_masks (np.ndarray): shape (N, H, W)
    """

    import os

    def mem(tag):
        rss = -1
        peak = -1

        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024**2
                elif line.startswith("VmPeak:"):
                    peak = int(line.split()[1]) / 1024**2

        print(
            f"\n[METRIC MEM] {tag}\n"
            f"  RSS    : {rss:.3f} GiB\n"
            f"  VmPeak : {peak:.3f} GiB",
            flush=True,
        )

    print(
        "\n========== calculate_metrics START ==========",
        flush=True,
    )

    print(
        f"scores: shape={scores.shape}, "
        f"dtype={scores.dtype}, "
        f"memory={scores.nbytes / 1024**3:.3f} GiB",
        flush=True,
    )

    print(
        f"gt_masks: shape={gt_masks.shape}, "
        f"dtype={gt_masks.dtype}, "
        f"memory={gt_masks.nbytes / 1024**3:.3f} GiB",
        flush=True,
    )

    mem("calculate_metrics entry")


    # # ========================================================
    # # Pixel AP
    # # ========================================================

    # print(
    #     "\n[METRIC 1] Pixel AP START",
    #     flush=True,
    # )
    # mem("before Pixel AP")

    # pix_ap = round(
    #     average_precision_score(
    #         gt_masks.flatten(),
    #         scores.flatten(),
    #     ),
    #     5,
    # )

    # print(
    #     f"[METRIC 1] Pixel AP DONE: {pix_ap}",
    #     flush=True,
    # )
    # mem("after Pixel AP")


    # # ========================================================
    # # Pixel F1 / precision-recall
    # # ========================================================

    # print(
    #     "\n[METRIC 2] precision_recall_curve START",
    #     flush=True,
    # )
    # mem("before precision_recall_curve")

    # precisions, recalls, _ = precision_recall_curve(
    #     gt_masks.flatten(),
    #     scores.flatten(),
    # )

    # print(
    #     "[METRIC 2] precision_recall_curve DONE",
    #     flush=True,
    # )

    # print(
    #     f"  precisions = {precisions.shape}, "
    #     f"{precisions.nbytes / 1024**3:.3f} GiB",
    #     flush=True,
    # )

    # print(
    #     f"  recalls = {recalls.shape}, "
    #     f"{recalls.nbytes / 1024**3:.3f} GiB",
    #     flush=True,
    # )

    # mem("after precision_recall_curve")


    # print(
    #     "\n[METRIC 3] Pixel F1 START",
    #     flush=True,
    # )

    # f1_scores = (
    #     2 * precisions * recalls
    # ) / (
    #     precisions + recalls
    # )

    # pix_f1_score = round(
    #     np.max(
    #         f1_scores[
    #             np.isfinite(f1_scores)
    #         ]
    #     ),
    #     5,
    # )

    # print(
    #     f"[METRIC 3] Pixel F1 DONE: "
    #     f"{pix_f1_score}",
    #     flush=True,
    # )

    # mem("after Pixel F1")


    # # ========================================================
    # # Pixel ROC AUC
    # # ========================================================

    # print(
    #     "\n[METRIC 4] Pixel ROC-AUC START",
    #     flush=True,
    # )

    # mem("before Pixel ROC-AUC")

    # pix_auc = round(
    #     roc_auc_score(
    #         gt_masks.flatten(),
    #         scores.flatten(),
    #     ),
    #     5,
    # )

    # print(
    #     f"[METRIC 4] Pixel ROC-AUC DONE: "
    #     f"{pix_auc}",
    #     flush=True,
    # )

    # mem("after Pixel ROC-AUC")
    # ============================================================
    # Pixel metric sampling
    # ============================================================

    pixel_scores = scores.reshape(-1)
    pixel_labels = gt_masks.reshape(-1)

    total_pixels = pixel_scores.shape[0]

    max_pixel_samples = 2_000_000

    if total_pixels > max_pixel_samples:
        rng = np.random.default_rng(42)

        sample_idx = rng.choice(
            total_pixels,
            size=max_pixel_samples,
            replace=False,
        )

        pixel_scores_eval = pixel_scores[sample_idx]
        pixel_labels_eval = pixel_labels[sample_idx]

    else:
        pixel_scores_eval = pixel_scores
        pixel_labels_eval = pixel_labels


    print(
        f"[Pixel Metrics] "
        f"{len(pixel_scores_eval):,} / "
        f"{total_pixels:,} pixels",
        flush=True,
    )


    # Pixel AP
    pix_ap = round(
        average_precision_score(
            pixel_labels_eval,
            pixel_scores_eval,
        ),
        5,
    )


    # Pixel F1
    precisions, recalls, _ = precision_recall_curve(
        pixel_labels_eval,
        pixel_scores_eval,
    )

    f1_scores = (
        2 * precisions * recalls
    ) / (
        precisions + recalls
    )

    pix_f1_score = round(
        np.max(
            f1_scores[
                np.isfinite(f1_scores)
            ]
        ),
        5,
    )


    # Pixel ROC-AUC
    pix_auc = round(
        roc_auc_score(
            pixel_labels_eval,
            pixel_scores_eval,
        ),
        5,
    )


    # ========================================================
    # Image metrics
    # ========================================================

    print(
        "\n[METRIC 5] Image metrics START",
        flush=True,
    )

    if img_scores is None:
        _, h, w = scores.shape
        size = h * w

        if only_max_value:
            topks = [1]
        else:
            topks = [
                int(size * p)
                for p in np.arange(
                    0.01,
                    0.41,
                    0.01,
                )
            ]

            topks = [1, 100] + topks

        img_aps = []
        img_aucs = []
        img_f1_scores = []

        for topk in topks:
            img_scores_ = get_image_scores(
                scores,
                topk,
            )

            img_ap = round(
                average_precision_score(
                    labels,
                    img_scores_,
                ),
                5,
            )

            precisions_img, recalls_img, _ = (
                precision_recall_curve(
                    labels,
                    img_scores_,
                )
            )

            f1_scores_img = (
                2
                * precisions_img
                * recalls_img
            ) / (
                precisions_img
                + recalls_img
            )

            img_f1_score = round(
                np.max(
                    f1_scores_img[
                        np.isfinite(
                            f1_scores_img
                        )
                    ]
                ),
                5,
            )

            img_auc = round(
                roc_auc_score(
                    labels,
                    img_scores_,
                ),
                5,
            )

            img_aps.append(img_ap)
            img_aucs.append(img_auc)
            img_f1_scores.append(
                img_f1_score
            )

        img_ap = np.max(img_aps)
        img_auc = np.max(img_aucs)
        img_f1_score = np.max(
            img_f1_scores
        )

    else:
        # img_ap = round(
        #     average_precision_score(
        #         labels,
        #         img_scores,
        #     ),
        #     5,
        # )

        # img_auc = round(
        #     roc_auc_score(
        #         labels,
        #         img_scores,
        #     ),
        #     5,
        # )

        # precisions_img, recalls_img, _ = (
        #     precision_recall_curve(
        #         labels,
        #         img_scores,
        #     )
        # )

        # f1_scores_img = (
        #     2
        #     * precisions_img
        #     * recalls_img
        # ) / (
        #     precisions_img
        #     + recalls_img
        # )

        # img_f1_score = round(
        #     np.max(
        #         f1_scores_img[
        #             np.isfinite(
        #                 f1_scores_img
        #             )
        #         ]
        #     ),
        #     5,
        # )
            img_ap = round(
                average_precision_score(
                    labels,
                    img_scores,
                ),
                5,
            )

            img_auc = round(
                roc_auc_score(
                    labels,
                    img_scores,
                ),
                5,
            )

            # ========================================================
            # Precision / Recall / Threshold
            # ========================================================

            precisions_img, recalls_img, thresholds_img = (
                precision_recall_curve(
                    labels,
                    img_scores,
                )
            )

            # thresholds_img has one fewer element than
            # precisions_img / recalls_img
            precisions_for_th = precisions_img[:-1]
            recalls_for_th = recalls_img[:-1]

            denominator = (
                precisions_for_th
                + recalls_for_th
            )

            f1_scores_for_th = np.divide(
                2
                * precisions_for_th
                * recalls_for_th,
                denominator,
                out=np.zeros_like(
                    denominator
                ),
                where=denominator != 0,
            )

            # ========================================================
            # Best F1 threshold
            # ========================================================

            best_idx = np.argmax(
                f1_scores_for_th
            )

            best_image_threshold = float(
                thresholds_img[
                    best_idx
                ]
            )

            img_f1_score = round(
                float(
                    f1_scores_for_th[
                        best_idx
                    ]
                ),
                5,
            )

            print()
            print(
                "============================================"
            )
            print(
                " Best Image-level Threshold"
            )
            print(
                "============================================"
            )
            print(
                f"Threshold : "
                f"{best_image_threshold:.6f}"
            )
            print(
                f"Best F1   : "
                f"{img_f1_score:.5f}"
            )
            print(
                "============================================"
            )

            # ========================================================
            # Save threshold for external inference
            # ========================================================

            os.makedirs(
                "./thresholds",
                exist_ok=True,
            )

            np.save(
                "./thresholds/"
                "patchcore_image_threshold.npy",
                np.array(
                    best_image_threshold,
                    dtype=np.float32,
                ),
            )

            print(
                "[Threshold] Saved to: "
                "./thresholds/"
                "patchcore_image_threshold.npy"
            )

    print(
        "[METRIC 5] Image metrics DONE",
        flush=True,
    )

    print(
        f"  Image AUC = {img_auc}\n"
        f"  Image AP  = {img_ap}\n"
        f"  Image F1  = {img_f1_score}",
        flush=True,
    )

    mem("after image metrics")


    # ========================================================
    # AUPRO
    # ========================================================

    print(
        f"\n[METRIC 6] AUPRO "
        f"pro={pro}",
        flush=True,
    )

    if pro:
        pix_aupro = calculate_aupro(
            gt_masks,
            scores,
        )
    else:
        pix_aupro = -1

    print(
        f"[METRIC 6] AUPRO DONE: "
        f"{pix_aupro}",
        flush=True,
    )

    mem("calculate_metrics end")

    print(
        "\n========== calculate_metrics DONE ==========",
        flush=True,
    )

    return (
        img_auc,
        img_ap,
        img_f1_score,
        pix_auc,
        pix_ap,
        pix_f1_score,
        pix_aupro,
    )


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