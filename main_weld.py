import os
import random
import warnings
import argparse

from PIL import Image
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ============================================================
# Dataset
# ============================================================

from datasets.weld import WeldAD
from datasets.mvtec import MVTEC

# ============================================================
# ADPretrain
# ============================================================

from validate import validate

from models.clip import ClipModel
from models.dino import DinoModel
from models.imagebind import ImageBindModel
from models.projector import MultiScaleAttentionProjector

from utils import (
    init_seeds,
    get_residual_features,
    get_mc_matched_ref_features,
    patchify_mask,
)

from losses.norm_loss import calculate_norm_loss
from losses.angle_loss import calculate_angle_loss


warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

TOTAL_SHOT = 8

# training 도중 reference normal image 수를 1 / 4 / 8 중 랜덤 선택
RANDOM_SHOT = True

# 한 이미지를 2개 augmentation view로 생성
WITH_AUG = True

# angle loss에서 center 사용
WITH_CENTER = True

DTYPE = torch.bfloat16

NORMALIZE = "imagenet"


# ============================================================
# Main
# ============================================================

def main(args):
    global NORMALIZE

    # Welding은 object class 하나
    # porosity / undercut / lack_of_fusion / incomplete_penetration
    # 은 anomaly type으로 취급
    CLASSES = ["weld"]

    # ========================================================
    # Normalization
    # ========================================================

    if args.backbone == "imagebind":
        NORMALIZE = "imagebind"
    elif "clip" in args.backbone:
        NORMALIZE = "imagebind"
    elif "dino" in args.backbone:
        NORMALIZE = "imagenet"
    else:
        NORMALIZE = "imagenet"

    # ========================================================
    # ADPretrain training dataset
    # ========================================================

    train_dataset = WeldAD(
        args.train_dataset_dir,
        class_name=None,
        train=True,
        normalize=NORMALIZE,
        with_aug=WITH_AUG,
        img_size=224,
        crp_size=224,
        msk_size=224,
        msk_crp_size=224,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    print()
    print("============================================")
    print(" ADPretrain Welding Training")
    print("============================================")
    print("Train dataset :", args.train_dataset_dir)
    print("Train samples :", len(train_dataset))
    print("Backbone      :", args.backbone)
    print("Batch size    :", args.batch_size)
    print("Epochs        :", args.epochs)
    print("============================================")
    print()

    # ========================================================
    # Backbone + Projector
    # ========================================================

    if args.backbone == "imagebind":
        encoder = ImageBindModel(
            args.backbone,
            device=args.device,
        )
        projectors = MultiScaleAttentionProjector(
            encoder.feature_dimensions,
            device=args.device,
        )

    elif "clip" in args.backbone:
        encoder = ClipModel(
            args.backbone,
            device=args.device,
        )
        projectors = MultiScaleAttentionProjector(
            encoder.feature_dimensions,
            device=args.device,
        )

    elif "dino" in args.backbone:
        encoder = DinoModel(
            args.backbone,
            device=args.device,
        )
        projectors = MultiScaleAttentionProjector(
            encoder.feature_dimensions,
            device=args.device,
        )

    else:
        raise ValueError(
            f"Unrecognized Backbone: {args.backbone}"
        )

    args.feature_levels = len(encoder.feature_dimensions)

    # Encoder는 고정 feature extractor로 사용
    encoder.to(DTYPE)
    encoder.to(args.device)
    encoder.eval()

    # 실제 학습 대상은 projector
    projectors.to(DTYPE)
    projectors.to(args.device)

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        projectors.parameters(),
        lr=args.lr,
        weight_decay=0.0005,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[7, 9],
        gamma=0.1,
    )

    # ========================================================
    # Weld center features
    # ========================================================

    centers = np.load(args.feature_centers)

    print(
        "Feature center shape:",
        centers.shape,
    )

    if centers.shape[0] != args.feature_levels:
        raise ValueError(
            f"Center feature levels mismatch: "
            f"{centers.shape[0]} != "
            f"{args.feature_levels}"
        )

    centers = torch.from_numpy(
        centers
    ).to(args.device)

    # ========================================================
    # Evaluation reference features
    #
    # center feature와는 다른 8-shot normal reference feature
    # ========================================================

    test_ref_features = load_mc_reference_features(
        args.test_ref_feature_dir,
        CLASSES,
        args.device,
        args.num_ref_shot,
    )

    print()
    print("============================================")
    print(" Evaluation Reference Features")
    print("============================================")

    for i, ref in enumerate(
        test_ref_features["weld"],
        start=1,
    ):
        print(
            f"Layer {i}: {tuple(ref.shape)}"
        )

    print("============================================")
    print()

    # ========================================================
    # Best metrics
    # ========================================================

    best_pro = 0.0
    best_img_auc = 0.0
    best_pix_auc = 0.0

    best_pro_norm = 0.0
    best_img_auc_norm = 0.0
    best_pix_auc_norm = 0.0

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(args.epochs):
        projectors.train()

        progress_bar = tqdm(
            total=len(train_loader)
        )

        progress_bar.set_description(
            f"Epoch[{epoch + 1}/{args.epochs}]"
        )

        for step, batch in enumerate(train_loader):
            loss_angle_total = 0.0
            loss_norm_total = 0.0

            num_angle = 0
            num_norm = 0

            (
                images,
                labels,
                masks,
                class_names,
                views,
            ) = batch

            # anomaly가 하나도 없는 batch는 skip
            if torch.sum(labels) == 0:
                progress_bar.update(1)
                continue

            # MoCoTransform이 반환한 두 augmentation view
            images1, images2 = images

            images1 = images1.to(
                dtype=DTYPE,
                device=args.device,
            )

            images2 = images2.to(
                dtype=DTYPE,
                device=args.device,
            )

            masks = masks.to(args.device)

            # =================================================
            # Frozen encoder feature
            # =================================================

            with torch.no_grad():
                features1 = encoder.encode_image_from_tensors(
                    images1
                )

                features2 = encoder.encode_image_from_tensors(
                    images2
                )

            # =================================================
            # Normal reference feature
            # =================================================

            ref_features = get_mc_reference_features(
                encoder,
                train_dataset,
                class_names,
                views,
                images1.device,
                args.train_ref_shot,
            )

            # =================================================
            # Residual features
            # =================================================

            mfeatures1 = get_mc_matched_ref_features(
                features1,
                class_names,
                ref_features,
            )

            rfeatures1 = get_residual_features(
                features1,
                mfeatures1,
            )

            mfeatures2 = get_mc_matched_ref_features(
                features2,
                class_names,
                ref_features,
            )

            rfeatures2 = get_residual_features(
                features2,
                mfeatures2,
            )

            # =================================================
            # Pixel mask -> feature-level mask
            # =================================================

            lvl_masks1 = []
            lvl_masks2 = []

            for l in range(args.feature_levels):
                _, _, h, w = rfeatures1[l].size()

                m1 = patchify_mask(
                    masks,
                    patch_size=224 // h,
                )

                lvl_masks1.append(m1)

                m2 = F.interpolate(
                    masks,
                    size=(h, w),
                    mode="nearest",
                ).squeeze(1)

                lvl_masks2.append(m2)

            # =================================================
            # ADPretrain projector
            # =================================================

            rfeatures1 = projectors(*rfeatures1)
            rfeatures2 = projectors(*rfeatures2)

            loss = None

            # =================================================
            # Multi-level loss
            # =================================================

            for l in range(args.feature_levels):
                e1 = rfeatures1[l]
                e2 = rfeatures2[l]

                _, dim, h, w = e1.size()

                e1 = (
                    e1
                    .permute(0, 2, 3, 1)
                    .reshape(-1, dim)
                )

                e2 = (
                    e2
                    .permute(0, 2, 3, 1)
                    .reshape(-1, dim)
                )

                m1 = lvl_masks1[l].reshape(-1)
                m2 = lvl_masks2[l].reshape(-1)

                # =============================================
                # Angle Loss
                # =============================================

                if WITH_CENTER:
                    center = centers[l]

                    loss_angle = calculate_angle_loss(
                        e1,
                        e2,
                        m1,
                        center,
                    )
                else:
                    loss_angle = calculate_angle_loss(
                        e1,
                        e2,
                        m1,
                    )

                if (
                    torch.is_tensor(loss_angle)
                    and (
                        torch.isnan(loss_angle)
                        or torch.isinf(loss_angle)
                    )
                ):
                    loss_angle = 0
                else:
                    if torch.is_tensor(loss_angle):
                        loss_angle_total += loss_angle.item()

                num_angle += 1

                # =============================================
                # Norm Loss
                # =============================================

                loss_norm1 = calculate_norm_loss(
                    e1,
                    m2,
                )

                loss_norm2 = calculate_norm_loss(
                    e2,
                    m2,
                )

                loss_norm = 0

                if torch.is_tensor(loss_norm1):
                    loss_norm += loss_norm1
                    loss_norm_total += loss_norm1.item()

                if torch.is_tensor(loss_norm2):
                    loss_norm += loss_norm2
                    loss_norm_total += loss_norm2.item()

                num_norm += 1

                # =============================================
                # Total loss
                # =============================================

                if (
                    isinstance(loss_angle, int)
                    and
                    isinstance(loss_norm, int)
                ):
                    continue

                loss_l = loss_angle + loss_norm

                if loss is None:
                    loss = loss_l
                else:
                    loss += loss_l

            # =================================================
            # backward
            # =================================================

            if loss is None:
                progress_bar.update(1)
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(
                angle_loss=(
                    f"{loss_angle_total / max(num_angle, 1):.3f}"
                ),
                norm_loss=(
                    f"{loss_norm_total / max(num_norm, 1):.3f}"
                ),
            )

            progress_bar.update(1)

        scheduler.step()
        progress_bar.close()

        # ====================================================
        # IMPORTANT:
        # Evaluation보다 먼저 training checkpoint 저장
        #
        # 앞으로 evaluation에서 에러가 나더라도
        # 해당 epoch의 projector weight를 잃지 않도록 함
        # ====================================================

        os.makedirs(
            args.checkpoint_path,
            exist_ok=True,
        )

        train_checkpoint = {
            "epoch": epoch + 1,
            "projectors": projectors.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "backbone": args.backbone,
        }

        train_checkpoint_path = os.path.join(
            args.checkpoint_path,
            f"weld_epoch_{epoch + 1:02d}_before_eval.pth",
        )

        torch.save(
            train_checkpoint,
            train_checkpoint_path,
        )

        print()
        print(
            "[Checkpoint] Saved training checkpoint:"
        )
        print(
            f"  {train_checkpoint_path}"
        )

        # ====================================================
        # Evaluation
        # ====================================================

        if (epoch + 1) % args.eval_freq == 0:
            print()
            print("============================================")
            print(
                f" Welding Evaluation - Epoch {epoch + 1}"
            )
            print("============================================")

            projectors.eval()

            # ------------------------------------------------
            # PatchCore normal memory bank
            #
            # expected:
            # test_dataset_dir/
            #   weld/
            #     train/good/
            # ------------------------------------------------

            normal_dataset = MVTEC(
                args.test_dataset_dir,
                class_name="weld",
                train=True,
                normalize=NORMALIZE,
                img_size=224,
                crp_size=224,
                msk_size=224,
                msk_crp_size=224,
            )

            # ------------------------------------------------
            # Evaluation dataset
            #
            # expected:
            # test_dataset_dir/
            #   weld/
            #     test/
            #       good/
            #       porosity/
            #       incomplete_penetration/
            #       lack_of_fusion/
            #       undercut/
            #     ground_truth/
            #       ...
            # ------------------------------------------------

            test_dataset = MVTEC(
                args.test_dataset_dir,
                class_name="weld",
                train=False,
                normalize=NORMALIZE,
                img_size=224,
                crp_size=224,
                msk_size=224,
                msk_crp_size=224,
            )

            print(
                "Normal eval images:",
                len(normal_dataset),
            )

            print(
                "Test images:",
                len(test_dataset),
            )

            normal_loader = DataLoader(
                normal_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
            )

            # ------------------------------------------------
            # Original ADPretrain validate()
            # ------------------------------------------------

            metrics = validate(
                args,
                encoder,
                projectors,
                normal_loader,
                test_loader,
                test_ref_features["weld"],
                args.device,
                "weld",
                residual=True,
                pro=True,
                dtype=DTYPE,
            )

            # =================================================
            # Angle / PatchCore metrics
            # =================================================

            (
                img_auc,
                img_ap,
                img_f1_score,
                pix_auc,
                pix_ap,
                pix_f1_score,
                pix_aupro,
            ) = metrics["metrics"]

            print()
            print("--------------------------------------------")
            print("[PatchCore / Angle]")
            print(f"Image AUC : {img_auc:.5f}")
            print(f"Image AP  : {img_ap:.5f}")
            print(f"Image F1  : {img_f1_score:.5f}")
            print(f"Pixel AUC : {pix_auc:.5f}")
            print(f"Pixel AP  : {pix_ap:.5f}")
            print(f"Pixel F1  : {pix_f1_score:.5f}")
            print(f"AUPRO     : {pix_aupro:.5f}")
            print("--------------------------------------------")

            state_dict = {
                "epoch": epoch + 1,
                "projectors": projectors.state_dict(),
                "img_auc": img_auc,
                "pix_auc": pix_auc,
                "pix_pro": pix_aupro,
                "backbone": args.backbone,
            }

            # Image AUC best
            if img_auc > best_img_auc:
                best_img_auc = img_auc

                torch.save(
                    state_dict,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_img_angle.pth",
                    ),
                )

                print(
                    "[SAVE] New best Image-AUC checkpoint"
                )

            # Pixel AUC best
            if pix_auc > best_pix_auc:
                best_pix_auc = pix_auc

                torch.save(
                    state_dict,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_pix_angle.pth",
                    ),
                )

                print(
                    "[SAVE] New best Pixel-AUC checkpoint"
                )

            # AUPRO best
            if pix_aupro > best_pro:
                best_pro = pix_aupro

                torch.save(
                    state_dict,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_pro_angle.pth",
                    ),
                )

                print(
                    "[SAVE] New best AUPRO checkpoint"
                )

            # =================================================
            # Norm metrics
            # =================================================

            (
                img_auc_norm,
                img_ap_norm,
                img_f1_norm,
                pix_auc_norm,
                pix_ap_norm,
                pix_f1_norm,
                pix_aupro_norm,
            ) = metrics["metrics-norm"]

            print()
            print("--------------------------------------------")
            print("[Feature Norm]")
            print(f"Image AUC : {img_auc_norm:.5f}")
            print(f"Image AP  : {img_ap_norm:.5f}")
            print(f"Image F1  : {img_f1_norm:.5f}")
            print(f"Pixel AUC : {pix_auc_norm:.5f}")
            print(f"Pixel AP  : {pix_ap_norm:.5f}")
            print(f"Pixel F1  : {pix_f1_norm:.5f}")
            print(f"AUPRO     : {pix_aupro_norm:.5f}")
            print("--------------------------------------------")

            state_dict_norm = {
                "epoch": epoch + 1,
                "projectors": projectors.state_dict(),
                "img_auc": img_auc_norm,
                "pix_auc": pix_auc_norm,
                "pix_pro": pix_aupro_norm,
                "backbone": args.backbone,
            }

            if img_auc_norm > best_img_auc_norm:
                best_img_auc_norm = img_auc_norm

                torch.save(
                    state_dict_norm,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_img_norm.pth",
                    ),
                )

            if pix_auc_norm > best_pix_auc_norm:
                best_pix_auc_norm = pix_auc_norm

                torch.save(
                    state_dict_norm,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_pix_norm.pth",
                    ),
                )

            if pix_aupro_norm > best_pro_norm:
                best_pro_norm = pix_aupro_norm

                torch.save(
                    state_dict_norm,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_pro_norm.pth",
                    ),
                )

            print()
            print("============================================")
            print(
                f"Best Image AUC : {best_img_auc:.5f}"
            )
            print(
                f"Best Pixel AUC : {best_pix_auc:.5f}"
            )
            print(
                f"Best AUPRO     : {best_pro:.5f}"
            )
            print("============================================")


# ============================================================
# Evaluation Reference Features
# ============================================================

def load_mc_reference_features(
    root_dir,
    class_names,
    device,
    num_shot=4,
):
    refs = {}

    for class_name in class_names:
        class_dir = os.path.join(
            root_dir,
            class_name,
        )

        layer1_refs = np.load(
            os.path.join(
                class_dir,
                "layer1.npy",
            )
        )

        layer2_refs = np.load(
            os.path.join(
                class_dir,
                "layer2.npy",
            )
        )

        layer3_refs = np.load(
            os.path.join(
                class_dir,
                "layer3.npy",
            )
        )

        layer4_refs = np.load(
            os.path.join(
                class_dir,
                "layer4.npy",
            )
        )

        layer1_refs = torch.from_numpy(
            layer1_refs
        ).to(
            dtype=DTYPE,
            device=device,
        )

        layer2_refs = torch.from_numpy(
            layer2_refs
        ).to(
            dtype=DTYPE,
            device=device,
        )

        layer3_refs = torch.from_numpy(
            layer3_refs
        ).to(
            dtype=DTYPE,
            device=device,
        )

        layer4_refs = torch.from_numpy(
            layer4_refs
        ).to(
            dtype=DTYPE,
            device=device,
        )

        K1 = (
            layer1_refs.shape[0]
            // TOTAL_SHOT
        ) * num_shot

        K2 = (
            layer2_refs.shape[0]
            // TOTAL_SHOT
        ) * num_shot

        K3 = (
            layer3_refs.shape[0]
            // TOTAL_SHOT
        ) * num_shot

        K4 = (
            layer4_refs.shape[0]
            // TOTAL_SHOT
        ) * num_shot

        layer1_refs = layer1_refs[:K1]
        layer2_refs = layer2_refs[:K2]
        layer3_refs = layer3_refs[:K3]
        layer4_refs = layer4_refs[:K4]

        refs[class_name] = (
            layer1_refs,
            layer2_refs,
            layer3_refs,
            layer4_refs,
        )

    return refs


# ============================================================
# Training Reference Features
# ============================================================

def get_mc_reference_features(
    encoder,
    dataset,
    class_names,
    views,
    device,
    num_shot=4,
):
    if RANDOM_SHOT:
        num_shot = random.choice(
            [1, 4, 8]
        )

    reference_features = {}

    for idx, (
        class_name,
        view,
    ) in enumerate(
        zip(
            class_names,
            views,
        )
    ):
        normal_paths = dataset.get_random_normal_images(
            class_name,
            view,
            num_shot,
        )

        images = load_and_transform_vision_data(
            normal_paths,
            device,
        )

        with torch.no_grad():
            features = encoder.encode_image_from_tensors(
                images.to(
                    dtype=DTYPE,
                    device=device,
                ),
                shape="seq",
            )

            for l in range(len(features)):
                _, _, c = features[l].shape

                features[l] = (
                    features[l]
                    .reshape(-1, c)
                )

            reference_features[idx] = features

    return reference_features


# ============================================================
# Reference image transform
# ============================================================

def load_and_transform_vision_data(
    image_paths,
    device,
):
    if NORMALIZE == "imagenet":
        mean = [
            0.485,
            0.456,
            0.406,
        ]

        std = [
            0.229,
            0.224,
            0.225,
        ]
    else:
        mean = (
            0.48145466,
            0.4578275,
            0.40821073,
        )

        std = (
            0.26862954,
            0.26130258,
            0.27577711,
        )

    if image_paths is None:
        return None

    image_outputs = []

    for image_path in image_paths:
        data_transform = T.Compose([
            T.Resize(
                224,
                T.InterpolationMode.BICUBIC,
            ),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=mean,
                std=std,
            ),
        ])

        with open(
            image_path,
            "rb",
        ) as fopen:
            image = Image.open(
                fopen
            ).convert("RGB")

        image = data_transform(
            image
        ).to(device)

        image_outputs.append(image)

    return torch.stack(
        image_outputs,
        dim=0,
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dataset_dir",
        type=str,
        default=(
            "/ssdg/spl_yejin/"
            "aihub_weld/"
            "weld_adpretrain"
        ),
    )

    parser.add_argument(
        "--test_dataset_dir",
        type=str,
        default=(
            "/ssdg/spl_yejin/"
            "aihub_weld/"
            "weld_eval_root"
        ),
    )

    parser.add_argument(
        "--test_ref_feature_dir",
        type=str,
        default=(
            "./ref_features/"
            "dinov2-base/"
            "weld_8shot"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=(
            "./checkpoints/"
            "dinov2-base-weld"
        ),
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2-base",
    )

    parser.add_argument(
        "--feature_centers",
        type=str,
        default=(
            "./centers/"
            "dino_base_weld_"
            "rfeature_centers.npy"
        ),
    )

    parser.add_argument(
        "--feature_levels",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--train_ref_shot",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num_ref_shot",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    init_seeds(42)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    main(args)