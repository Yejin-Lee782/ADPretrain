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
from utils import calculate_metrics, init_seeds
from ad_models.patchcore import get_patchcore

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as l2_normalize

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

def save_anomaly_maps(
    pix_scores,
    gt_masks,
    gt_labels,
    image_paths,
    class_name,
    output_dir="./visualizations/heatmap",
):
    output_dir = Path(output_dir) / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, anomaly_map in enumerate(pix_scores):

        anomaly_map = np.squeeze(anomaly_map)
        gt_mask = np.squeeze(gt_masks[i])
        label = int(gt_labels[i])

        image_path = image_paths[i]

        # DataLoader collate 때문에 tuple/list일 수 있음
        if isinstance(image_path, (list, tuple)):
            image_path = image_path[0]

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WARNING] Failed to load image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # anomaly map을 원본 이미지 크기로 확대
        anomaly_map_resized = cv2.resize(
            anomaly_map,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        # GT mask도 혹시 크기가 다르면 맞춰줌
        gt_mask_resized = cv2.resize(
            gt_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        # heatmap 시각화용 normalize
        vmin = anomaly_map_resized.min()
        vmax = np.percentile(anomaly_map_resized, 99)

        normalized_map = (anomaly_map_resized - vmin) / (vmax - vmin + 1e-8)
        normalized_map = np.clip(normalized_map, 0, 1)

        heatmap_gray = (normalized_map * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        overlay_rgb = cv2.addWeighted(
            image_rgb,
            0.6,
            heatmap_rgb,
            0.4,
            0,
        )

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(gt_mask_resized, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        im = axes[2].imshow(anomaly_map_resized, cmap="jet")
        axes[2].set_title(f"Anomaly Map | label={label}")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(overlay_rgb)
        axes[3].set_title("Overlay")
        axes[3].axis("off")

        plt.tight_layout()

        save_path = output_dir / f"{i:04d}_label_{label}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Heatmaps saved to: {output_dir}")

def prepare_pca_features(
    train_features,
    test_features,
    patch_shapes,
    gt_labels,
    gt_masks,
):
    """
    PatchCore feature를 다음 4개 그룹으로 분리

    1. Train Normal
    2. Test Normal
    3. Defect image 안의 실제 defect region
    4. Defect image 안의 normal region
    """

    train_normal = np.asarray(train_features)

    test_normal_list = []
    defect_list = []
    defect_background_list = []

    for features, patch_shape, label, gt_mask in zip(
        test_features,
        patch_shapes,
        gt_labels,
        gt_masks,
    ):

        features = np.asarray(features)

        # patch_shape = [H_patch, W_patch]
        ph, pw = patch_shape

        # -------------------------------------------------
        # 정상 이미지
        # -------------------------------------------------
        if not bool(label):

            test_normal_list.append(
                features
            )

            continue

        # -------------------------------------------------
        # 불량 이미지
        # -------------------------------------------------
        gt_mask = np.squeeze(
            gt_mask
        ).astype(np.uint8)

        # GT pixel mask를 patch grid 크기로 축소
        patch_mask = cv2.resize(
            gt_mask,
            (pw, ph),
            interpolation=cv2.INTER_NEAREST,
        )

        patch_mask = (
            patch_mask.reshape(-1) > 0
        )

        # 안전 확인
        if len(patch_mask) != len(features):

            raise ValueError(
                f"Patch mask/features mismatch: "
                f"{len(patch_mask)} vs {len(features)}, "
                f"patch_shape={patch_shape}, "
                f"features.shape={features.shape}"
            )

        # 실제 defect가 포함된 patch
        if patch_mask.any():

            defect_list.append(
                features[patch_mask]
            )

        # defect image이지만 GT상 정상 영역인 patch
        if (~patch_mask).any():

            defect_background_list.append(
                features[~patch_mask]
            )

    # -------------------------------------------------
    # concatenate
    # -------------------------------------------------

    test_normal = (
        np.concatenate(
            test_normal_list,
            axis=0
        )
        if len(test_normal_list) > 0
        else None
    )

    defect = (
        np.concatenate(
            defect_list,
            axis=0
        )
        if len(defect_list) > 0
        else None
    )

    defect_background = (
        np.concatenate(
            defect_background_list,
            axis=0
        )
        if len(defect_background_list) > 0
        else None
    )

    return (
        train_normal,
        test_normal,
        defect,
        defect_background,
    )


def visualize_3d_pca(
    train_normal,
    test_normal,
    defect,
    defect_background=None,
    max_samples=3000,
    save_path="./visualizations/pca_3d.png",
):
    """
    PatchCore feature space를 3차원 PCA로 시각화
    """

    rng = np.random.default_rng(42)

    def preprocess(features):

        if features is None:
            return None

        features = np.asarray(features)

        if len(features) == 0:
            return None

        # PCA plot이 너무 무거워지는 것 방지
        if len(features) > max_samples:

            indices = rng.choice(
                len(features),
                max_samples,
                replace=False,
            )

            features = features[indices]

        # embedding 방향 비교를 위해 L2 normalization
        return l2_normalize(
            features,
            norm="l2"
        )

    train_normal = preprocess(
        train_normal
    )

    test_normal = preprocess(
        test_normal
    )

    defect = preprocess(
        defect
    )

    defect_background = preprocess(
        defect_background
    )

    groups = []
    group_names = []

    if train_normal is not None:

        groups.append(
            train_normal
        )

        group_names.append(
            "Train Normal"
        )

    if test_normal is not None:

        groups.append(
            test_normal
        )

        group_names.append(
            "Test Normal"
        )

    if defect_background is not None:

        groups.append(
            defect_background
        )

        group_names.append(
            "Defect Image - Normal Region"
        )

    if defect is not None:

        groups.append(
            defect
        )

        group_names.append(
            "Actual Defect Region"
        )

    if len(groups) < 2:

        raise RuntimeError(
            "PCA requires at least two non-empty feature groups."
        )

    # -------------------------------------------------
    # 모든 group을 같은 PCA 공간에서 fit
    # -------------------------------------------------

    all_features = np.concatenate(
        groups,
        axis=0
    )

    print(
        "[PCA] Input feature shape:",
        all_features.shape
    )

    pca = PCA(
        n_components=3
    )

    reduced = pca.fit_transform(
        all_features
    )

    print(
        "[PCA] Explained variance ratio:",
        pca.explained_variance_ratio_
    )

    print(
        "[PCA] Total explained variance:",
        pca.explained_variance_ratio_.sum()
    )

    # -------------------------------------------------
    # plot
    # -------------------------------------------------

    fig = plt.figure(
        figsize=(11, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    start = 0

    for features, name in zip(
        groups,
        group_names
    ):

        end = start + len(features)

        xyz = reduced[
            start:end
        ]

        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=12,
            alpha=0.5,
            label=name,
        )

        start = end

    explained = (
        pca.explained_variance_ratio_
        * 100
    )

    ax.set_xlabel(
        f"PC1 ({explained[0]:.1f}%)"
    )

    ax.set_ylabel(
        f"PC2 ({explained[1]:.1f}%)"
    )

    ax.set_zlabel(
        f"PC3 ({explained[2]:.1f}%)"
    )

    ax.set_title(
        "ADPretrain + PatchCore Feature Space"
    )

    ax.legend()

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[PCA] Saved to: {save_path}"
    )

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
    encoder.eval()
    
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
    elif args.dataset == 'weld':
        CLASS_NAMES = ['weld']
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset}!")
    
    # if args.with_pretrained:
    #     ref_features = load_mc_reference_features(args.ref_feature_dir, CLASS_NAMES, args.device, num_shot=args.num_ref_shot)
    # else:
    #     ref_features = None
    if args.with_pretrained:
        ref_features = load_mc_reference_features(
            args.ref_feature_dir,
            CLASS_NAMES,
            args.device,
            num_shot=args.num_ref_shot,
        )
    else:
        ref_features = None
    
    img_aucs, img_aps, img_f1_scores, pix_aucs, pix_aps, pix_f1_scores, pix_aupros = [], [], [], [], [], [], []
    for class_name in CLASS_NAMES:
        print()
        print(
            f"Reference features for [{class_name}]"
        )

        for i, ref in enumerate(
            ref_features[class_name],
            start=1
        ):
            print(
                f"  Layer {i}: {ref.shape}"
            )
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
        elif args.dataset == 'weld':

            train_dataset = MVTEC(args.dataset_dir, class_name=class_name, train=True, normalize=normalize, img_size=224, crp_size=224, msk_size=224, msk_crp_size=224,
            )

            test_dataset = MVTEC(
                args.dataset_dir,
                class_name=class_name,
                train=False,
                normalize=normalize,
                img_size=224,
                crp_size=224,
                msk_size=224,
                msk_crp_size=224,
            )
        else:
            raise ValueError(f"Unrecognized dataset {args.dataset}!")
        
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
        
        patchcore_model = get_patchcore(encoder, projectors, device=args.device, residual=args.with_pretrained)
        patchcore_model.fit(train_loader, ref_features[class_name] if args.with_pretrained else None, class_name, aligned=False)
        # img_scores, pix_scores, gt_labels, gt_masks = patchcore_model.predict(
        #     test_loader, ref_features[class_name] if args.with_pretrained else None, class_name, aligned=False
        # )
        (
            img_scores,
            pix_scores,
            gt_labels,
            gt_masks,
            test_features,
            patch_shapes,
            returned_metadata,
        ) = patchcore_model.predict(
            test_loader,
            ref_features[class_name]
            if args.with_pretrained
            else None,
            class_name,
            aligned=False,
            return_features=True,
        )
        # 실제 test image path
        image_paths = test_dataset.image_paths

        print(
            "[Visualization] Number of image paths:",
            len(image_paths)
        )

        print(
            "[Visualization] First image paths:"
        )

        for path in image_paths[:3]:
            print(" ", path)
        img_scores = np.array(img_scores)
        pix_scores = np.asarray(pix_scores)
        gt_labels = np.concatenate(gt_labels)
        gt_masks = np.concatenate(gt_masks, axis=0)

        # -----------------------------
        # visualization
        # -----------------------------
        save_anomaly_maps(
            pix_scores=pix_scores,
            gt_masks=gt_masks,
            gt_labels=gt_labels,
            image_paths=image_paths,
            class_name=class_name,
        )
        # ======================================================
        # PCA Feature Preparation
        # ======================================================

        print()
        print("Preparing PCA features...")

        (
            train_normal_features,
            test_normal_features,
            defect_features,
            defect_background_features,
        ) = prepare_pca_features(

            # PatchCore training memory bank
            train_features=(
                patchcore_model.memory_bank_features
            ),

            # test image patch features
            test_features=test_features,

            # 각 test image patch grid
            patch_shapes=patch_shapes,

            gt_labels=gt_labels,

            gt_masks=gt_masks,
        )


        # ======================================================
        # Shape 확인
        # ======================================================

        print(
            "[PCA] Train Normal:",
            train_normal_features.shape
        )


        if test_normal_features is not None:

            print(
                "[PCA] Test Normal:",
                test_normal_features.shape
            )


        if defect_features is not None:

            print(
                "[PCA] Actual Defect:",
                defect_features.shape
            )


        if defect_background_features is not None:

            print(
                "[PCA] Defect Image Normal Region:",
                defect_background_features.shape
            )


        # ======================================================
        # 3D PCA
        # ======================================================

        visualize_3d_pca(
            train_normal=train_normal_features,
            test_normal=test_normal_features,
            defect=defect_features,
            defect_background=defect_background_features,
            max_samples=3000,
            save_path=(
                f"./visualizations/"
                f"{class_name}/pca_3d.png"
            ),
        )
        
        img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(img_scores, pix_scores, gt_labels, gt_masks, pro=True, only_max_value=True)
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
    #parser.add_argument('--with_pretrained', default=True, action="store_true")
    parser.add_argument('--with_pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pretrained_weights', type=str, default="./checkpoints/dinov2-large/checkpoints_pro_angle.pth")
    
    parser.add_argument('--dataset', type=str, default="weld")
    parser.add_argument('--dataset_dir', type=str, default="./data/weld")
    parser.add_argument('--ref_feature_dir', type=str, default="./ref_features/dinov2-large/mvtec_8shot")
    
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--num_ref_shot", type=int, default=8)
    
    args = parser.parse_args()
    init_seeds(42)
    
    main(args)
