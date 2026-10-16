import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def save_heatmaps(
    pix_scores,
    gt_masks,
    gt_labels,
    image_paths,
    output_dir="./visualizations/heatmap",
):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for i in range(
        len(pix_scores)
    ):

        anomaly_map = np.squeeze(
            pix_scores[i]
        )

        gt_mask = np.squeeze(
            gt_masks[i]
        )

        label = int(
            gt_labels[i]
        )

        image_path = image_paths[i]

        # DataLoader가 path를 tuple 등으로 반환하는 경우
        if isinstance(
            image_path,
            (list, tuple)
        ):
            image_path = image_path[0]

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"[WARNING] Cannot load: {image_path}"
            )
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        h, w = image.shape[:2]

        # anomaly map resize
        resized_map = cv2.resize(
            anomaly_map,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

        # percentile normalization
        vmin = resized_map.min()

        vmax = np.percentile(
            resized_map,
            99
        )

        normalized = (
            resized_map - vmin
        ) / (
            vmax - vmin + 1e-8
        )

        normalized = np.clip(
            normalized,
            0,
            1
        )

        heatmap = (
            normalized * 255
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        overlay = cv2.addWeighted(
            image,
            0.6,
            heatmap,
            0.4,
            0,
        )

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(18, 4)
        )

        axes[0].imshow(
            image
        )
        axes[0].set_title(
            "Original"
        )

        axes[1].imshow(
            gt_mask,
            cmap="gray"
        )
        axes[1].set_title(
            "GT Mask"
        )

        im = axes[2].imshow(
            resized_map,
            cmap="jet"
        )
        axes[2].set_title(
            "Anomaly Map"
        )

        fig.colorbar(
            im,
            ax=axes[2],
            fraction=0.046,
            pad=0.04
        )

        axes[3].imshow(
            overlay
        )
        axes[3].set_title(
            "Overlay"
        )

        for ax in axes:
            ax.axis(
                "off"
            )

        fig.suptitle(
            f"Label: {'Defect' if label else 'Normal'}"
        )

        plt.tight_layout()

        save_path = os.path.join(
            output_dir,
            f"{i:04d}_label_{label}.png"
        )

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(
            fig
        )

    print(
        f"[Visualization] Heatmaps saved: {output_dir}"
    )