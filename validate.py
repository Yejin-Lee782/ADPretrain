import json
import warnings
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F

from utils import get_residual_features, get_matched_ref_features
from utils import calculate_metrics, load_and_transform_vision_data
from ad_models.patchcore import get_patchcore
from datasets.mvtec import MVTEC
from datasets.visa import VISA
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD

warnings.filterwarnings('ignore')
import os
import gc


def print_memory(tag):
    rss_gb = -1
    peak_gb = -1
    swap_gb = -1

    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_gb = int(line.split()[1]) / 1024**2
                elif line.startswith("VmPeak:"):
                    peak_gb = int(line.split()[1]) / 1024**2
                elif line.startswith("VmSwap:"):
                    swap_gb = int(line.split()[1]) / 1024**2
    except Exception:
        pass

    if torch.cuda.is_available():
        cuda_alloc = (
            torch.cuda.memory_allocated()
            / 1024**3
        )
        cuda_reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )
    else:
        cuda_alloc = 0
        cuda_reserved = 0

    print(
        f"\n[MEM] {tag}\n"
        f"  CPU RSS      : {rss_gb:.3f} GiB\n"
        f"  CPU VmPeak   : {peak_gb:.3f} GiB\n"
        f"  CPU Swap     : {swap_gb:.3f} GiB\n"
        f"  CUDA allocated: {cuda_alloc:.3f} GiB\n"
        f"  CUDA reserved : {cuda_reserved:.3f} GiB",
        flush=True,
    )


def print_array(name, x):
    if isinstance(x, np.ndarray):
        print(
            f"[ARRAY] {name}: "
            f"shape={x.shape}, "
            f"dtype={x.dtype}, "
            f"size={x.size:,}, "
            f"memory={x.nbytes / 1024**3:.3f} GiB",
            flush=True,
        )

    elif torch.is_tensor(x):
        nbytes = (
            x.numel()
            * x.element_size()
        )

        print(
            f"[TENSOR] {name}: "
            f"shape={tuple(x.shape)}, "
            f"dtype={x.dtype}, "
            f"device={x.device}, "
            f"memory={nbytes / 1024**3:.3f} GiB",
            flush=True,
        )

# def validate(args, encoder, projectors, normal_loader, test_loader, ref_features, device, class_name, residual=True, pro=False, dtype=torch.float):
#     if projectors is not None:
#         projectors.eval()
    
#     patchcore_model = get_patchcore(encoder, projectors, device=device, dtype=dtype, residual=residual)
#     patchcore_model.fit(normal_loader, ref_features, class_name, aligned=False)
#     img_scores, pix_scores, gt_labels, gt_masks = patchcore_model.predict(
#         test_loader, ref_features, class_name, aligned=False
#     )
#     img_scores = np.array(img_scores)
#     pix_scores = np.asarray(pix_scores)
#     gt_labels = np.concatenate(gt_labels)
#     gt_masks = np.concatenate(gt_masks, axis=0)
#     img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(img_scores, pix_scores, gt_labels, gt_masks, pro=pro, only_max_value=True)
    
#     metrics_norm = calculate_scores_by_feature_norm(args, encoder, projectors, test_loader, ref_features, device, class_name, 
#                                                     residual=residual, pro=pro, dtype=dtype)
#     # metrics_norm = None
#     img_auc_norm, img_ap_norm, img_f1_score_norm, pix_auc_norm, pix_ap_norm, pix_f1_score_norm, pix_aupro_norm = metrics_norm
#     metrics = {}
#     metrics['metrics'] = [img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro]
#     metrics['metrics-norm'] = [img_auc_norm, img_ap_norm, img_f1_score_norm, pix_auc_norm, pix_ap_norm, pix_f1_score_norm, pix_aupro_norm]
    
#     return metrics
def validate(
    args,
    encoder,
    projectors,
    normal_loader,
    test_loader,
    ref_features,
    device,
    class_name,
    residual=True,
    pro=False,
    dtype=torch.float,
):
    print("\n========== validate START ==========", flush=True)
    print_memory("validate START")

    if projectors is not None:
        projectors.eval()

    print("\n[1] Creating PatchCore model", flush=True)

    patchcore_model = get_patchcore(
        encoder,
        projectors,
        device=device,
        dtype=dtype,
        residual=residual,
    )

    print_memory("after get_patchcore")


    # ========================================================
    # PatchCore normal memory bank
    # ========================================================

    print(
        "\n[2] PatchCore.fit START",
        flush=True,
    )

    patchcore_model.fit(
        normal_loader,
        ref_features,
        class_name,
        aligned=False,
    )

    print(
        "[2] PatchCore.fit DONE",
        flush=True,
    )

    print_memory(
        "after PatchCore.fit"
    )


    # ========================================================
    # PatchCore test inference
    # ========================================================

    print(
        "\n[3] PatchCore.predict START",
        flush=True,
    )

    img_scores, pix_scores, gt_labels, gt_masks = (
        patchcore_model.predict(
            test_loader,
            ref_features,
            class_name,
            aligned=False,
        )
    )

    print(
        "[3] PatchCore.predict DONE",
        flush=True,
    )

    print(
        f"  len(img_scores) = "
        f"{len(img_scores)}",
        flush=True,
    )

    print(
        f"  len(pix_scores) = "
        f"{len(pix_scores)}",
        flush=True,
    )

    print(
        f"  len(gt_labels) = "
        f"{len(gt_labels)}",
        flush=True,
    )

    print(
        f"  len(gt_masks) = "
        f"{len(gt_masks)}",
        flush=True,
    )

    print_memory(
        "after PatchCore.predict"
    )


    # ========================================================
    # Convert to contiguous arrays
    # ========================================================

    print(
        "\n[4] np.array(img_scores)",
        flush=True,
    )

    img_scores = np.array(img_scores)

    print_array(
        "img_scores",
        img_scores,
    )

    print_memory("after img_scores np.array")
    print("\n[5] np.asarray(pix_scores)", flush=True,)

    pix_scores = np.asarray(pix_scores)

    print_array("pix_scores", pix_scores,)
    print_memory("after pix_scores np.asarray")
    print("\n[6] concatenate gt_labels", flush=True,)

    gt_labels = np.concatenate(
        gt_labels
    )

    print_array(
        "gt_labels",
        gt_labels,
    )

    print_memory(
        "after gt_labels concatenate"
    )


    print(
        "\n[7] concatenate gt_masks",
        flush=True,
    )

    gt_masks = np.concatenate(
        gt_masks,
        axis=0,
    )

    print_array(
        "gt_masks",
        gt_masks,
    )

    print_memory(
        "after gt_masks concatenate"
    )

    # ========================================================
    # First metric calculation
    # ========================================================

    print(
        "\n[8] PatchCore calculate_metrics START",
        flush=True,
    )

    print_memory(
        "before first calculate_metrics"
    )

    (
        img_auc,
        img_ap,
        img_f1_score,
        pix_auc,
        pix_ap,
        pix_f1_score,
        pix_aupro,
    ) = calculate_metrics(
        img_scores,
        pix_scores,
        gt_labels,
        gt_masks,
        pro=pro,
        only_max_value=True,
    )

    print(
        "[8] PatchCore calculate_metrics DONE",
        flush=True,
    )

    print(
        f"  Image AUC = {img_auc}\n"
        f"  Image AP  = {img_ap}\n"
        f"  Image F1  = {img_f1_score}\n"
        f"  Pixel AUC = {pix_auc}\n"
        f"  Pixel AP  = {pix_ap}\n"
        f"  Pixel F1  = {pix_f1_score}\n"
        f"  AUPRO     = {pix_aupro}",
        flush=True,
    )

    print_memory(
        "after first calculate_metrics"
    )


    # ========================================================
    # Second evaluation: feature norm
    # ========================================================

    print(
        "\n========================================",
        flush=True,
    )
    print(
        "[9] Feature-Norm evaluation START",
        flush=True,
    )
    print(
        "========================================",
        flush=True,
    )

    print_memory(
        "BEFORE calculate_scores_by_feature_norm"
    )


    metrics_norm = calculate_scores_by_feature_norm(
        args,
        encoder,
        projectors,
        test_loader,
        ref_features,
        device,
        class_name,
        residual=residual,
        pro=pro,
        dtype=dtype,
    )


    print(
        "[9] Feature-Norm evaluation DONE",
        flush=True,
    )

    print_memory(
        "AFTER calculate_scores_by_feature_norm"
    )


    (
        img_auc_norm,
        img_ap_norm,
        img_f1_score_norm,
        pix_auc_norm,
        pix_ap_norm,
        pix_f1_score_norm,
        pix_aupro_norm,
    ) = metrics_norm


    metrics = {}

    metrics["metrics"] = [
        img_auc,
        img_ap,
        img_f1_score,
        pix_auc,
        pix_ap,
        pix_f1_score,
        pix_aupro,
    ]

    metrics["metrics-norm"] = [
        img_auc_norm,
        img_ap_norm,
        img_f1_score_norm,
        pix_auc_norm,
        pix_ap_norm,
        pix_f1_score_norm,
        pix_aupro_norm,
    ]

    print(
        "\n========== validate DONE ==========",
        flush=True,
    )

    return metrics


# def calculate_scores_by_feature_norm(args, encoder, projectors, test_loader, ref_features, device, class_name, residual=True, aligned=False, pro=False, dtype=torch.float):
#     if aligned:
#         if class_name in MVTEC.CLASS_NAMES:
#             with open(f'./aligned/references/mvtec/{class_name}.json', 'r', encoding='utf-8') as f:
#                 matched_reference_paths = json.load(f)
#         elif class_name in VISA.CLASS_NAMES:
#             with open(f'./aligned/references/visa/{class_name}.json', 'r', encoding='utf-8') as f:
#                 matched_reference_paths = json.load(f)
#         elif class_name in BTAD.CLASS_NAMES:
#             with open(f'./aligned/references/btad/{class_name}.json', 'r', encoding='utf-8') as f:
#                 matched_reference_paths = json.load(f)
#         elif class_name in MVTEC3D.CLASS_NAMES:
#             with open(f'./aligned/references/mvtec3d/{class_name}.json', 'r', encoding='utf-8') as f:
#                 matched_reference_paths = json.load(f)
#         elif class_name in MPDD.CLASS_NAMES:
#             with open(f'./aligned/references/mpdd/{class_name}.json', 'r', encoding='utf-8') as f:
#                 matched_reference_paths = json.load(f)
                
#     label_list, gt_mask_list = [], []
#     s_list = [list() for _ in range(args.feature_levels)]
#     progress_bar = tqdm(total=len(test_loader))
#     progress_bar.set_description(f"Evaluating")
#     for idx, batch in enumerate(test_loader):
#         progress_bar.update(1)
        
#         image, label, mask, _ = batch    
#         gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
#         label_list.append(label.cpu().numpy().astype(bool).ravel())
        
#         image = image.to(dtype=dtype, device=device)
#         size = image.shape[-1]
        
#         if aligned:
#             image_path = test_loader.dataset.image_paths[idx]
#             rimage_paths = matched_reference_paths[image_path]
#             rimages = load_and_transform_vision_data(rimage_paths, device, encoder.name)
#             with torch.no_grad():
#                 features = encoder.encode_image_from_tensors(rimages.to(dtype=dtype, device=device), shape='seq')
#                 for l in range(len(features)):
#                     _, _, c = features[l].shape
#                     features[l] = features[l].reshape(-1, c)
#             ref_features = features
        
#         with torch.no_grad():
#             features = encoder.encode_image_from_tensors(image, shape='img')
            
#             if residual:
#                 mfeatures = get_matched_ref_features(features, ref_features)
#                 rfeatures = get_residual_features(features, mfeatures)
#                 if projectors is not None:
#                     features = projectors(*rfeatures)
#                 else:
#                     features = rfeatures
#             else:
#                 if projectors is not None:
#                     features = projectors(*features)
            
#             for l in range(len(features)):
#                 e = features[l]  
#                 bs, dim, h, w = e.size()
#                 e = e.permute(0, 2, 3, 1).reshape(-1, dim)
#                 s = e.norm(dim=1)
#                 s = torch.sqrt(s + 1) - 1
#                 s = s.to(torch.float)

#                 s_list[l].append(s.reshape(bs, h, w))
    
#     progress_bar.close()
    
#     labels = np.concatenate(label_list)
#     gt_masks = np.concatenate(gt_mask_list, axis=0)
    
#     scores = aggregate_anomaly_scores(s_list, feature_levels=args.feature_levels, class_name=class_name, size=size)
#     img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(None, scores, labels, gt_masks, pro=pro, only_max_value=True)

#     return img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro

def calculate_scores_by_feature_norm(
    args,
    encoder,
    projectors,
    test_loader,
    ref_features,
    device,
    class_name,
    residual=True,
    aligned=False,
    pro=False,
    dtype=torch.float,
):
    print(
        "\n========== Feature Norm START ==========",
        flush=True,
    )

    print_memory(
        "Feature Norm START"
    )

    if aligned:
        if class_name in MVTEC.CLASS_NAMES:
            with open(
                f"./aligned/references/mvtec/{class_name}.json",
                "r",
                encoding="utf-8",
            ) as f:
                matched_reference_paths = json.load(f)

        elif class_name in VISA.CLASS_NAMES:
            with open(
                f"./aligned/references/visa/{class_name}.json",
                "r",
                encoding="utf-8",
            ) as f:
                matched_reference_paths = json.load(f)

        elif class_name in BTAD.CLASS_NAMES:
            with open(
                f"./aligned/references/btad/{class_name}.json",
                "r",
                encoding="utf-8",
            ) as f:
                matched_reference_paths = json.load(f)

        elif class_name in MVTEC3D.CLASS_NAMES:
            with open(
                f"./aligned/references/mvtec3d/{class_name}.json",
                "r",
                encoding="utf-8",
            ) as f:
                matched_reference_paths = json.load(f)

        elif class_name in MPDD.CLASS_NAMES:
            with open(
                f"./aligned/references/mpdd/{class_name}.json",
                "r",
                encoding="utf-8",
            ) as f:
                matched_reference_paths = json.load(f)


    label_list = []
    gt_mask_list = []

    s_list = [
        list()
        for _ in range(
            args.feature_levels
        )
    ]


    progress_bar = tqdm(
        total=len(test_loader)
    )

    progress_bar.set_description(
        "Evaluating"
    )


    for idx, batch in enumerate(
        test_loader
    ):
        progress_bar.update(1)

        image, label, mask, _ = batch

        gt_mask_list.append(
            mask
            .squeeze(1)
            .cpu()
            .numpy()
            .astype(bool)
        )

        label_list.append(
            label
            .cpu()
            .numpy()
            .astype(bool)
            .ravel()
        )

        image = image.to(
            dtype=dtype,
            device=device,
        )

        size = image.shape[-1]


        if aligned:
            image_path = (
                test_loader
                .dataset
                .image_paths[idx]
            )

            rimage_paths = (
                matched_reference_paths[
                    image_path
                ]
            )

            rimages = (
                load_and_transform_vision_data(
                    rimage_paths,
                    device,
                    encoder.name,
                )
            )

            with torch.no_grad():
                features = (
                    encoder
                    .encode_image_from_tensors(
                        rimages.to(
                            dtype=dtype,
                            device=device,
                        ),
                        shape="seq",
                    )
                )

                for l in range(
                    len(features)
                ):
                    _, _, c = (
                        features[l].shape
                    )

                    features[l] = (
                        features[l]
                        .reshape(-1, c)
                    )

            ref_features = features


        with torch.no_grad():
            features = (
                encoder
                .encode_image_from_tensors(
                    image,
                    shape="img",
                )
            )


            if residual:
                mfeatures = (
                    get_matched_ref_features(
                        features,
                        ref_features,
                    )
                )

                rfeatures = (
                    get_residual_features(
                        features,
                        mfeatures,
                    )
                )

                if projectors is not None:
                    features = projectors(
                        *rfeatures
                    )
                else:
                    features = rfeatures

            else:
                if projectors is not None:
                    features = projectors(
                        *features
                    )


            for l in range(
                len(features)
            ):
                e = features[l]

                bs, dim, h, w = (
                    e.size()
                )

                e = (
                    e
                    .permute(0, 2, 3, 1)
                    .reshape(-1, dim)
                )

                s = e.norm(dim=1)

                s = (
                    torch.sqrt(s + 1)
                    - 1
                )

                s = s.to(torch.float)

                s_list[l].append(
                    s.reshape(
                        bs,
                        h,
                        w,
                    )
                )


        # ====================================================
        # Print every 500 images
        # ====================================================

        if (
            (idx + 1) % 500 == 0
            or idx == 0
        ):
            print(
                f"\n[NORM] processed "
                f"{idx + 1}/"
                f"{len(test_loader)}",
                flush=True,
            )

            print(
                "  s_list lengths:",
                [
                    len(x)
                    for x in s_list
                ],
                flush=True,
            )

            total_s_bytes = 0

            for level_items in s_list:
                for tensor in level_items:
                    total_s_bytes += (
                        tensor.numel()
                        * tensor.element_size()
                    )

            print(
                f"  s_list total = "
                f"{total_s_bytes / 1024**3:.3f} GiB",
                flush=True,
            )

            print_memory(
                f"FeatureNorm image {idx + 1}"
            )


    progress_bar.close()


    print(
        "\n[NORM] Inference loop DONE",
        flush=True,
    )

    print_memory(
        "after FeatureNorm inference loop"
    )


    # ========================================================
    # Labels / masks
    # ========================================================

    print(
        "\n[NORM] concatenate labels",
        flush=True,
    )

    labels = np.concatenate(
        label_list
    )

    print_array(
        "norm labels",
        labels,
    )

    print_memory(
        "after norm labels"
    )


    print(
        "\n[NORM] concatenate GT masks",
        flush=True,
    )

    gt_masks = np.concatenate(
        gt_mask_list,
        axis=0,
    )

    print_array(
        "norm gt_masks",
        gt_masks,
    )

    print_memory(
        "after norm gt_masks"
    )


    # ========================================================
    # Aggregation
    # ========================================================

    print(
        "\n[NORM] aggregate_anomaly_scores START",
        flush=True,
    )

    scores = aggregate_anomaly_scores(
        s_list,
        feature_levels=args.feature_levels,
        class_name=class_name,
        size=size,
    )

    print(
        "[NORM] aggregate_anomaly_scores DONE",
        flush=True,
    )

    print_array(
        "norm scores",
        scores,
    )

    print_memory(
        "after aggregate_anomaly_scores"
    )


    # ========================================================
    # Second metric computation
    # ========================================================

    print(
        "\n[NORM] second calculate_metrics START",
        flush=True,
    )

    print_memory(
        "before second calculate_metrics"
    )


    (
        img_auc,
        img_ap,
        img_f1_score,
        pix_auc,
        pix_ap,
        pix_f1_score,
        pix_aupro,
    ) = calculate_metrics(
        None,
        scores,
        labels,
        gt_masks,
        pro=pro,
        only_max_value=True,
    )


    print(
        "[NORM] second calculate_metrics DONE",
        flush=True,
    )

    print_memory(
        "after second calculate_metrics"
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

# def aggregate_anomaly_scores(logps_list, feature_levels=3, class_name=None, size=224):
#     abnormal_map = [list() for _ in range(feature_levels)]
#     for l in range(feature_levels):
#         probs = torch.cat(logps_list[l], dim=0)  
#         # upsample
#         abnormal_map[l] = F.interpolate(probs.unsqueeze(1),
#             size=size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
#     # score aggregation
#     scores = np.zeros_like(abnormal_map[0])
#     for l in range(feature_levels):
#         scores += abnormal_map[l]
#     scores /= feature_levels
    
#     for i in range(scores.shape[0]):
#         scores[i] = gaussian_filter(scores[i], sigma=4)

#     return scores
def aggregate_anomaly_scores(
    logps_list,
    feature_levels=3,
    class_name=None,
    size=224,
):
    print(
        "\n========== aggregate_anomaly_scores ==========",
        flush=True,
    )

    print_memory(
        "aggregate START"
    )


    abnormal_map = [
        list()
        for _ in range(
            feature_levels
        )
    ]


    for l in range(
        feature_levels
    ):
        print(
            f"\n[AGG] Level {l} torch.cat START",
            flush=True,
        )

        print_memory(
            f"before level {l} cat"
        )


        probs = torch.cat(
            logps_list[l],
            dim=0,
        )


        print_array(
            f"level {l} probs",
            probs,
        )

        print_memory(
            f"after level {l} cat"
        )


        print(
            f"[AGG] Level {l} interpolate START",
            flush=True,
        )


        upsampled = F.interpolate(
            probs.unsqueeze(1),
            size=size,
            mode="bilinear",
            align_corners=True,
        ).squeeze()


        print_array(
            f"level {l} upsampled",
            upsampled,
        )

        print_memory(
            f"after level {l} interpolate"
        )


        print(
            f"[AGG] Level {l} GPU -> CPU numpy",
            flush=True,
        )


        abnormal_map[l] = (
            upsampled
            .cpu()
            .numpy()
        )


        print_array(
            f"abnormal_map[{l}]",
            abnormal_map[l],
        )

        print_memory(
            f"after level {l} CPU numpy"
        )


        del probs
        del upsampled

        torch.cuda.empty_cache()

        print_memory(
            f"after level {l} cleanup"
        )


    print(
        "\n[AGG] Creating scores",
        flush=True,
    )


    scores = np.zeros_like(
        abnormal_map[0]
    )

    print_array(
        "scores initial",
        scores,
    )

    print_memory(
        "after scores allocation"
    )


    for l in range(
        feature_levels
    ):
        print(
            f"[AGG] adding level {l}",
            flush=True,
        )

        scores += abnormal_map[l]

        print_memory(
            f"after adding level {l}"
        )


    scores /= feature_levels


    print(
        "[AGG] Gaussian filtering START",
        flush=True,
    )


    for i in range(
        scores.shape[0]
    ):
        scores[i] = gaussian_filter(
            scores[i],
            sigma=4,
        )

        if (
            (i + 1) % 1000 == 0
        ):
            print(
                f"[AGG] Gaussian "
                f"{i + 1}/"
                f"{scores.shape[0]}",
                flush=True,
            )

            print_memory(
                f"gaussian {i + 1}"
            )


    print(
        "[AGG] Gaussian filtering DONE",
        flush=True,
    )

    print_memory(
        "aggregate DONE"
    )


    return scores