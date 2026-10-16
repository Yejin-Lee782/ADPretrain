import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset

from datasets.mvtec import MVTEC
from models.dino import DinoModel
from models.projector import MultiScaleAttentionProjector
from ad_models.patchcore import get_patchcore

from main_weld_ddp import (
    DTYPE,
    NORMALIZE,
    load_mc_reference_features,
)

from eval_weld_checkpoint import (
    load_projector_checkpoint,
)


# ============================================================
# Single external image dataset
# ============================================================

class SingleExternalImageDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path

        self.transform = T.Compose([
            T.Resize(
                224,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.open(
            self.image_path
        ).convert("RGB")

        image = self.transform(
            image
        )

        # Dummy values.
        # GT label / GT mask are unknown
        label = torch.tensor(
            0,
            dtype=torch.long,
        )

        mask = torch.zeros(
            1,
            224,
            224,
            dtype=torch.float32,
        )

        class_name = "weld"

        return (
            image,
            label,
            mask,
            class_name,
        )


# ============================================================
# Visualization
# ============================================================

def save_external_heatmap(
    image_path,
    anomaly_map,
    output_path,
):
    image = cv2.imread(
        image_path
    )

    if image is None:
        raise RuntimeError(
            f"Cannot load image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    anomaly_map = np.squeeze(
        anomaly_map
    )

    h, w = image.shape[:2]

    resized_map = cv2.resize(
        anomaly_map,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    # --------------------------------------------------------
    # Heatmap normalization for visualization only
    # --------------------------------------------------------

    vmin = resized_map.min()

    vmax = np.percentile(
        resized_map,
        99,
    )

    normalized = (
        resized_map - vmin
    ) / (
        vmax - vmin + 1e-8
    )

    normalized = np.clip(
        normalized,
        0,
        1,
    )

    heatmap = (
        normalized * 255
    ).astype(
        np.uint8
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    axes[0].imshow(
        image
    )
    axes[0].set_title(
        "External Image"
    )

    im = axes[1].imshow(
        resized_map,
        cmap="jet",
    )
    axes[1].set_title(
        "Anomaly Map"
    )

    fig.colorbar(
        im,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )

    axes[2].imshow(
        overlay
    )
    axes[2].set_title(
        "Overlay"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[Visualization] Saved: "
        f"{output_path}"
    )


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device(
        args.device
    )

    torch.cuda.set_device(
        device
    )

    print()
    print(
        "============================================"
    )
    print(
        " External Weld Image Inference"
    )
    print(
        "============================================"
    )
    print(
        "Image      :",
        args.image_path,
    )
    print(
        "Checkpoint :",
        args.checkpoint,
    )
    print(
        "Device     :",
        device,
    )
    print(
        "============================================"
    )

    # ========================================================
    # Encoder
    # ========================================================

    encoder = DinoModel(
        args.backbone,
        device=str(device),
    )

    encoder.to(DTYPE)
    encoder.to(device)
    encoder.eval()

    # ========================================================
    # Projector
    # ========================================================

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
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    # ========================================================
    # Reference features
    # ========================================================

    ref_features = load_mc_reference_features(
        args.test_ref_feature_dir,
        ["weld"],
        device,
        args.num_ref_shot,
    )

    # ========================================================
    # Normal images for PatchCore memory bank
    # ========================================================

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

    if (
        args.max_normal_images > 0
        and len(normal_dataset)
        > args.max_normal_images
    ):
        rng = np.random.default_rng(
            42
        )

        indices = rng.choice(
            len(normal_dataset),
            size=args.max_normal_images,
            replace=False,
        )

        normal_dataset = Subset(
            normal_dataset,
            indices.tolist(),
        )

    normal_loader = DataLoader(
        normal_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(
        "Normal memory-bank images:",
        len(normal_dataset),
    )

    # ========================================================
    # External image
    # ========================================================

    external_dataset = (
        SingleExternalImageDataset(
            args.image_path
        )
    )

    external_loader = DataLoader(
        external_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # ========================================================
    # PatchCore
    # ========================================================

    patchcore_model = get_patchcore(
        encoder,
        projectors,
        device=device,
        dtype=DTYPE,
        residual=True,
    )

    print(
        "\n[1] Building PatchCore memory bank..."
    )

    patchcore_model.fit(
        normal_loader,
        ref_features["weld"],
        "weld",
        aligned=False,
    )

    print(
        "[2] Running external image..."
    )

    (
        img_scores,
        pix_scores,
        _,
        _,
    ) = patchcore_model.predict(
        external_loader,
        ref_features["weld"],
        "weld",
        aligned=False,
    )

    image_score = float(
        np.asarray(
            img_scores
        ).reshape(-1)[0]
    )

    anomaly_map = np.asarray(
        pix_scores
    )[0]

    # ========================================================
    # Load validation threshold
    # ========================================================

    if not os.path.exists(
        args.threshold_path
    ):
        raise FileNotFoundError(
            f"Threshold file not found: "
            f"{args.threshold_path}"
        )

    threshold = float(
        np.asarray(
            np.load(
                args.threshold_path
            )
        ).reshape(-1)[0]
    )


    # ========================================================
    # Normal / Defect prediction
    # ========================================================

    if image_score >= threshold:
        prediction = "Defect"
    else:
        prediction = "Normal"

    print()
    print(
        "============================================"
    )
    print(
        " External Image Result"
    )
    print(
        "============================================"
    )

    print(
        f"Image score : "
        f"{image_score:.6f}"
    )

    print(
        f"Threshold   : "
        f"{threshold:.6f}"
    )

    print(
        f"Prediction  : "
        f"{prediction}"
    )

    print(
        f"Score gap   : "
        f"{image_score - threshold:+.6f}"
    )

    print(
        "Anomaly map :",
        anomaly_map.shape,
    )

    print(
        "============================================"
    )

    save_external_heatmap(
        image_path=args.image_path,
        anomaly_map=anomaly_map,
        output_path=args.output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--threshold_path",
        type=str,
        default=(
            "./thresholds/"
            "patchcore_image_threshold.npy"
        ),
        help=(
            "Image-level anomaly threshold "
            "calculated from validation set"
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
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max_normal_images",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "./visualizations/"
            "external_test.png"
        ),
    )

    args = parser.parse_args()

    main(args)