import os
import random
import warnings
import argparse

from PIL import Image
from tqdm import tqdm

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.weld import WeldAD
from datasets.mvtec import MVTEC

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
from datetime import timedelta


warnings.filterwarnings("ignore")

TOTAL_SHOT = 8
RANDOM_SHOT = True
WITH_AUG = True
WITH_CENTER = True
DTYPE = torch.bfloat16
NORMALIZE = "imagenet"


def setup_distributed():
    required = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
    missing = [k for k in required if k not in os.environ]
    if missing:
        raise RuntimeError(
            "Run this script with torchrun. "
            f"Missing environment variables: {missing}"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP training.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=12),
    )

    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def distributed_all_true(local_condition, device):
    """Return True only when the condition is True on every rank."""
    flag = torch.tensor(
        1 if local_condition else 0,
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def main(args):
    global NORMALIZE

    rank, local_rank, world_size, device = setup_distributed()
    main_process = is_main_process(rank)

    # Existing ADPretrain helpers expect args.device.
    args.device = str(device)

    init_seeds(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        CLASSES = ["weld"]

        if args.backbone == "imagebind":
            NORMALIZE = "imagebind"
        elif "clip" in args.backbone:
            NORMALIZE = "imagebind"
        elif "dino" in args.backbone:
            NORMALIZE = "imagenet"
        else:
            NORMALIZE = "imagenet"

        # ====================================================
        # Training dataset + DistributedSampler
        # ====================================================
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

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,  # per GPU
            sampler=train_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if main_process:
            print()
            print("============================================")
            print(" ADPretrain Welding DDP Training")
            print("============================================")
            print("Train dataset       :", args.train_dataset_dir)
            print("Train samples       :", len(train_dataset))
            print("Backbone            :", args.backbone)
            print("World size          :", world_size)
            print("Visible GPUs        :", torch.cuda.device_count())
            print("Batch size / GPU    :", args.batch_size)
            print("Effective batch     :", args.batch_size * world_size)
            print("Epochs              :", args.epochs)
            print("Steps / GPU / epoch :", len(train_loader))
            print("============================================")
            print()

        # ====================================================
        # Encoder + Projector
        # ====================================================
        if args.backbone == "imagebind":
            encoder = ImageBindModel(args.backbone, device=args.device)
            projectors = MultiScaleAttentionProjector(
                encoder.feature_dimensions,
                device=args.device,
            )
        elif "clip" in args.backbone:
            encoder = ClipModel(args.backbone, device=args.device)
            projectors = MultiScaleAttentionProjector(
                encoder.feature_dimensions,
                device=args.device,
            )
        elif "dino" in args.backbone:
            encoder = DinoModel(args.backbone, device=args.device)
            projectors = MultiScaleAttentionProjector(
                encoder.feature_dimensions,
                device=args.device,
            )
        else:
            raise ValueError(f"Unrecognized Backbone: {args.backbone}")

        args.feature_levels = len(encoder.feature_dimensions)

        # Encoder is frozen.
        encoder.to(DTYPE)
        encoder.to(device)
        encoder.eval()

        # Only projector is trainable.
        projectors.to(DTYPE)
        projectors.to(device)

        # ====================================================
        # Optional resume: load projector before DDP wrapping
        # ====================================================
        resume_checkpoint = None
        start_epoch = 0

        best_pro = 0.0
        best_img_auc = 0.0
        best_pix_auc = 0.0
        best_pro_norm = 0.0
        best_img_auc_norm = 0.0
        best_pix_auc_norm = 0.0

        if args.resume is not None:
            resume_checkpoint = torch.load(args.resume, map_location="cpu")

            if "projectors" not in resume_checkpoint:
                raise KeyError(
                    "Resume checkpoint does not contain 'projectors'."
                )

            projectors.load_state_dict(
                resume_checkpoint["projectors"],
                strict=True,
            )

            start_epoch = int(resume_checkpoint.get("epoch", 0))
            best_pro = float(resume_checkpoint.get("best_pro", 0.0))
            best_img_auc = float(
                resume_checkpoint.get("best_img_auc", 0.0)
            )
            best_pix_auc = float(
                resume_checkpoint.get("best_pix_auc", 0.0)
            )
            best_pro_norm = float(
                resume_checkpoint.get("best_pro_norm", 0.0)
            )
            best_img_auc_norm = float(
                resume_checkpoint.get("best_img_auc_norm", 0.0)
            )
            best_pix_auc_norm = float(
                resume_checkpoint.get("best_pix_auc_norm", 0.0)
            )

        # ====================================================
        # DDP wrap
        # ====================================================
        projectors = DDP(
            projectors,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

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

        if resume_checkpoint is not None:
            if "optimizer" in resume_checkpoint:
                optimizer.load_state_dict(resume_checkpoint["optimizer"])
                optimizer_to(optimizer, device)

            if "scheduler" in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint["scheduler"])

            if main_process:
                print()
                print("[Resume] Loaded checkpoint:")
                print(" ", args.resume)
                print("[Resume] Completed epochs:", start_epoch)
                print("[Resume] Next epoch:", start_epoch + 1)
                print()

        # ====================================================
        # Feature centers
        # ====================================================
        centers = np.load(args.feature_centers)

        if main_process:
            print("Feature center shape:", centers.shape)

        if centers.shape[0] != args.feature_levels:
            raise ValueError(
                "Center feature levels mismatch: "
                f"{centers.shape[0]} != {args.feature_levels}"
            )

        centers = torch.from_numpy(centers).to(device)

        # ====================================================
        # Evaluation 8-shot references: rank 0 only
        # ====================================================
        test_ref_features = None

        if main_process:
            test_ref_features = load_mc_reference_features(
                args.test_ref_feature_dir,
                CLASSES,
                device,
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
                print(f"Layer {i}: {tuple(ref.shape)}")

            print("============================================")
            print()

        dist.barrier()

        if start_epoch >= args.epochs:
            if main_process:
                print(
                    "Nothing to train: "
                    f"checkpoint epoch={start_epoch}, "
                    f"--epochs={args.epochs}"
                )
            return

        # ====================================================
        # Training loop
        # ====================================================
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            projectors.train()

            progress_bar = tqdm(
                total=len(train_loader),
                disable=not main_process,
            )

            if main_process:
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

                # ------------------------------------------------
                # DDP safety:
                # if any rank has an all-normal batch, all ranks
                # skip this training step together.
                # ------------------------------------------------
                local_has_anomaly = torch.sum(labels).item() > 0
                all_have_anomaly = distributed_all_true(
                    local_has_anomaly,
                    device,
                )

                if not all_have_anomaly:
                    if main_process:
                        progress_bar.update(1)
                    continue

                images1, images2 = images

                images1 = images1.to(
                    dtype=DTYPE,
                    device=device,
                    non_blocking=True,
                )
                images2 = images2.to(
                    dtype=DTYPE,
                    device=device,
                    non_blocking=True,
                )
                masks = masks.to(
                    device,
                    non_blocking=True,
                )

                # Frozen encoder feature extraction
                with torch.no_grad():
                    features1 = encoder.encode_image_from_tensors(images1)
                    features2 = encoder.encode_image_from_tensors(images2)

                # Normal reference feature for the current batch
                ref_features = get_mc_reference_features(
                    encoder,
                    train_dataset,
                    class_names,
                    views,
                    device,
                    args.train_ref_shot,
                )

                # Residual features
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

                # Pixel mask -> feature-level mask
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

                # DDP-wrapped trainable projector
                rfeatures1 = projectors(*rfeatures1)
                rfeatures2 = projectors(*rfeatures2)

                loss = None

                for l in range(args.feature_levels):
                    e1 = rfeatures1[l]
                    e2 = rfeatures2[l]

                    _, dim, h, w = e1.size()

                    e1 = (
                        e1.permute(0, 2, 3, 1)
                        .reshape(-1, dim)
                    )
                    e2 = (
                        e2.permute(0, 2, 3, 1)
                        .reshape(-1, dim)
                    )

                    m1 = lvl_masks1[l].reshape(-1)
                    m2 = lvl_masks2[l].reshape(-1)

                    # Angle loss
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
                    elif torch.is_tensor(loss_angle):
                        loss_angle_total += loss_angle.item()

                    num_angle += 1

                    # Norm loss
                    loss_norm1 = calculate_norm_loss(e1, m2)
                    loss_norm2 = calculate_norm_loss(e2, m2)

                    loss_norm = 0

                    if torch.is_tensor(loss_norm1):
                        loss_norm += loss_norm1
                        loss_norm_total += loss_norm1.item()

                    if torch.is_tensor(loss_norm2):
                        loss_norm += loss_norm2
                        loss_norm_total += loss_norm2.item()

                    num_norm += 1

                    if (
                        isinstance(loss_angle, int)
                        and isinstance(loss_norm, int)
                    ):
                        continue

                    loss_l = loss_angle + loss_norm

                    if loss is None:
                        loss = loss_l
                    else:
                        loss += loss_l

                # DDP safety for loss=None
                all_have_loss = distributed_all_true(
                    loss is not None,
                    device,
                )

                if not all_have_loss:
                    if main_process:
                        progress_bar.update(1)
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if main_process:
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

            if main_process:
                progress_bar.close()

            # =================================================
            # Save training state BEFORE evaluation (rank 0 only)
            # =================================================
            if main_process:
                os.makedirs(args.checkpoint_path, exist_ok=True)

                train_checkpoint = {
                    "epoch": epoch + 1,
                    "projectors": unwrap_model(projectors).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "backbone": args.backbone,
                    "world_size": world_size,
                    "batch_size_per_gpu": args.batch_size,
                    "effective_batch_size": args.batch_size * world_size,
                    "best_pro": best_pro,
                    "best_img_auc": best_img_auc,
                    "best_pix_auc": best_pix_auc,
                    "best_pro_norm": best_pro_norm,
                    "best_img_auc_norm": best_img_auc_norm,
                    "best_pix_auc_norm": best_pix_auc_norm,
                }

                train_checkpoint_path = os.path.join(
                    args.checkpoint_path,
                    f"weld_epoch_{epoch + 1:02d}_before_eval.pth",
                )

                torch.save(train_checkpoint, train_checkpoint_path)

                print()
                print("[Checkpoint] Saved training checkpoint:")
                print(" ", train_checkpoint_path)

            # All ranks must reach the same synchronization point.
            dist.barrier()

            # =================================================
            # Evaluation: rank 0 only
            # =================================================
            if (epoch + 1) % args.eval_freq == 0:
                if main_process:
                    print()
                    print("============================================")
                    print(f" Welding Evaluation - Epoch {epoch + 1}")
                    print("============================================")

                    # Use the raw module, NOT the DDP wrapper, because
                    # rank 1 is waiting and validate() runs only on rank 0.
                    eval_projector = unwrap_model(projectors)
                    eval_projector.eval()

                    torch.cuda.empty_cache()

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

                    print("Normal eval images:", len(normal_dataset))
                    print("Test images:", len(test_dataset))

                    normal_loader = DataLoader(
                        normal_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.eval_num_workers,
                        pin_memory=True,
                        drop_last=False,
                    )

                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.eval_num_workers,
                        pin_memory=True,
                        drop_last=False,
                    )

                    metrics = validate(
                        args,
                        encoder,
                        eval_projector,
                        normal_loader,
                        test_loader,
                        test_ref_features["weld"],
                        device,
                        "weld",
                        residual=True,
                        pro=True,
                        dtype=DTYPE,
                    )

                    # -----------------------------------------
                    # PatchCore / Angle metrics
                    # -----------------------------------------
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
                        "projectors": eval_projector.state_dict(),
                        "img_auc": img_auc,
                        "pix_auc": pix_auc,
                        "pix_pro": pix_aupro,
                        "backbone": args.backbone,
                    }

                    if img_auc > best_img_auc:
                        best_img_auc = img_auc
                        torch.save(
                            state_dict,
                            os.path.join(
                                args.checkpoint_path,
                                "checkpoints_img_angle.pth",
                            ),
                        )
                        print("[SAVE] New best Image-AUC checkpoint")

                    if pix_auc > best_pix_auc:
                        best_pix_auc = pix_auc
                        torch.save(
                            state_dict,
                            os.path.join(
                                args.checkpoint_path,
                                "checkpoints_pix_angle.pth",
                            ),
                        )
                        print("[SAVE] New best Pixel-AUC checkpoint")

                    if pix_aupro > best_pro:
                        best_pro = pix_aupro
                        torch.save(
                            state_dict,
                            os.path.join(
                                args.checkpoint_path,
                                "checkpoints_pro_angle.pth",
                            ),
                        )
                        print("[SAVE] New best AUPRO checkpoint")

                    # -----------------------------------------
                    # Feature norm metrics
                    # -----------------------------------------
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
                        "projectors": eval_projector.state_dict(),
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
                    print(f"Best Image AUC : {best_img_auc:.5f}")
                    print(f"Best Pixel AUC : {best_pix_auc:.5f}")
                    print(f"Best AUPRO     : {best_pro:.5f}")
                    print("============================================")

                # rank 1 waits here while rank 0 evaluates.
                dist.barrier()
            else:
                dist.barrier()

    finally:
        cleanup_distributed()


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
        class_dir = os.path.join(root_dir, class_name)

        layer1_refs = np.load(os.path.join(class_dir, "layer1.npy"))
        layer2_refs = np.load(os.path.join(class_dir, "layer2.npy"))
        layer3_refs = np.load(os.path.join(class_dir, "layer3.npy"))
        layer4_refs = np.load(os.path.join(class_dir, "layer4.npy"))

        layer1_refs = torch.from_numpy(layer1_refs).to(
            dtype=DTYPE,
            device=device,
        )
        layer2_refs = torch.from_numpy(layer2_refs).to(
            dtype=DTYPE,
            device=device,
        )
        layer3_refs = torch.from_numpy(layer3_refs).to(
            dtype=DTYPE,
            device=device,
        )
        layer4_refs = torch.from_numpy(layer4_refs).to(
            dtype=DTYPE,
            device=device,
        )

        K1 = (layer1_refs.shape[0] // TOTAL_SHOT) * num_shot
        K2 = (layer2_refs.shape[0] // TOTAL_SHOT) * num_shot
        K3 = (layer3_refs.shape[0] // TOTAL_SHOT) * num_shot
        K4 = (layer4_refs.shape[0] // TOTAL_SHOT) * num_shot

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
        num_shot = random.choice([1, 4, 8])

    reference_features = {}

    for idx, (class_name, view) in enumerate(
        zip(class_names, views)
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
                features[l] = features[l].reshape(-1, c)

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
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
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
            T.Normalize(mean=mean, std=std),
        ])

        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_outputs.append(image)

    return torch.stack(image_outputs, dim=0)


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

    # Per-GPU batch size.
    # 2 GPUs * 8 = effective batch size 16.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    # Total target epochs.
    # If resume checkpoint completed epoch 1 and --epochs 3,
    # only epochs 2 and 3 are executed.
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=(
            "./checkpoints/"
            "dinov2-base-weld-ddp"
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

    # Per-rank workers. With two ranks, 4 means 8 workers total.
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    # Evaluation is rank 0 only.
    parser.add_argument(
        "--eval_num_workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    main(args)