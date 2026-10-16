import argparse
import torch
from torch.utils.data import DataLoader, Subset

from datasets.mvtec import MVTEC
from validate import validate
from models.dino import DinoModel
from models.projector import MultiScaleAttentionProjector

from main_weld_ddp import DTYPE, NORMALIZE, load_mc_reference_features


def load_projector_checkpoint(projectors, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "projectors" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'projectors'.")

    state_dict = checkpoint["projectors"]

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    projectors.load_state_dict(state_dict, strict=True)
    return checkpoint


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print()
    print("============================================")
    print(" Weld Checkpoint Evaluation")
    print("============================================")
    print("Checkpoint :", args.checkpoint)
    print("Backbone   :", args.backbone)
    print("Device     :", device)
    print("============================================")
    print()

    encoder = DinoModel(
        args.backbone,
        device=str(device),
    )
    encoder.to(DTYPE)
    encoder.to(device)
    encoder.eval()

    args.feature_levels = len(encoder.feature_dimensions)

    projectors = MultiScaleAttentionProjector(
        encoder.feature_dimensions,
        device=str(device),
    )
    projectors.to(DTYPE)
    projectors.to(device)

    checkpoint = load_projector_checkpoint(
        projectors,
        args.checkpoint,
    )
    projectors.eval()

    print(
        "Loaded checkpoint epoch:",
        checkpoint.get("epoch", "unknown"),
    )
    print(
        "Checkpoint backbone:",
        checkpoint.get("backbone", "unknown"),
    )

    if "effective_batch_size" in checkpoint:
        print(
            "Training effective batch size:",
            checkpoint["effective_batch_size"],
        )

    print()

    test_ref_features = load_mc_reference_features(
        args.test_ref_feature_dir,
        ["weld"],
        device,
        args.num_ref_shot,
    )

    print("Reference features:")
    for i, ref in enumerate(
        test_ref_features["weld"],
        start=1,
    ):
        print(f"  layer{i}: {tuple(ref.shape)}")

    print()

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
    if args.max_normal_images > 0:
        n = min(
            args.max_normal_images,
            len(normal_dataset),
        )

        generator = torch.Generator().manual_seed(42)

        indices = torch.randperm(
            len(normal_dataset),
            generator=generator,
        )[:n].tolist()

        normal_dataset = Subset(
            normal_dataset,
            indices,
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

    print("Normal memory-bank images:", len(normal_dataset))
    print("Test images:", len(test_dataset))

    normal_loader = DataLoader(
        normal_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    torch.cuda.empty_cache()

    metrics = validate(
        args,
        encoder,
        projectors,
        normal_loader,
        test_loader,
        test_ref_features["weld"],
        device,
        "weld",
        residual=True,
        pro=True,
        dtype=DTYPE,
    )

    (
        img_auc,
        img_ap,
        img_f1,
        pix_auc,
        pix_ap,
        pix_f1,
        pix_aupro,
    ) = metrics["metrics"]

    print()
    print("============================================")
    print(" [PatchCore / Angle]")
    print("============================================")
    print(f"Image AUC : {img_auc:.5f}")
    print(f"Image AP  : {img_ap:.5f}")
    print(f"Image F1  : {img_f1:.5f}")
    print(f"Pixel AUC : {pix_auc:.5f}")
    print(f"Pixel AP  : {pix_ap:.5f}")
    print(f"Pixel F1  : {pix_f1:.5f}")
    print(f"AUPRO     : {pix_aupro:.5f}")

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
    print("============================================")
    print(" [Feature Norm]")
    print("============================================")
    print(f"Image AUC : {img_auc_norm:.5f}")
    print(f"Image AP  : {img_ap_norm:.5f}")
    print(f"Image F1  : {img_f1_norm:.5f}")
    print(f"Pixel AUC : {pix_auc_norm:.5f}")
    print(f"Pixel AP  : {pix_ap_norm:.5f}")
    print(f"Pixel F1  : {pix_f1_norm:.5f}")
    print(f"AUPRO     : {pix_aupro_norm:.5f}")
    print("============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument(
        "--test_dataset_dir",
        type=str,
        default="/ssdg/spl_yejin/aihub_weld/weld_eval_root",
    )

    parser.add_argument(
        "--test_ref_feature_dir",
        type=str,
        default="./ref_features/dinov2-base/weld_8shot",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2-base",
    )

    parser.add_argument(
        "--num_ref_shot",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--feature_levels",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )

    # Extra fields kept for compatibility with validate(args, ...).
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--train_ref_shot", type=int, default=4)

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/dinov2-base-weld-ddp",
    )

    parser.add_argument(
        "--feature_centers",
        type=str,
        default="./centers/dino_base_weld_rfeature_centers.npy",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument(
        "--max_normal_images",
        type=int,
        default=5000,
    )

    args = parser.parse_args()
    main(args)