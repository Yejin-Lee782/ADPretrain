import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# RealIAD가 사용하는 augmentation을 그대로 사용
from datasets.real_iad import MoCoTransform


class WeldAD(Dataset):

    CLASS_NAMES = ["weld"]

    DEFECT_TYPES = [
        "porosity",
        "incomplete_penetration",
        "lack_of_fusion",
        "undercut",
    ]

    def __init__(
        self,
        root: str,
        class_name: Optional[str] = None,
        train: bool = True,
        normalize: str = "imagenet",
        with_aug: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs,
    ):

        self.root = Path(root)

        self.images_root = self.root / "images"
        self.masks_root = self.root / "masks"

        # Weld dataset에서는 object class가 하나뿐
        self.class_name = "weld"

        self.train = train

        self.cropsize = [
            kwargs.get("msk_crp_size", 224),
            kwargs.get("msk_crp_size", 224),
        ]

        # ----------------------------
        # Image transform
        # ----------------------------
        if normalize == "imagenet":

            self.transform = T.Compose([
                T.Resize(
                    kwargs.get("img_size", 224),
                    T.InterpolationMode.BICUBIC,
                ),

                T.CenterCrop(
                    kwargs.get("crp_size", 224)
                ),

                T.ToTensor(),

                T.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])

        else:

            self.transform = T.Compose([
                T.Resize(
                    kwargs.get("img_size", 224),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),

                T.CenterCrop(
                    kwargs.get("crp_size", 224)
                ),

                T.ToTensor(),

                T.Normalize(
                    mean=(
                        0.48145466,
                        0.4578275,
                        0.40821073,
                    ),
                    std=(
                        0.26862954,
                        0.26130258,
                        0.27577711,
                    ),
                ),
            ])

        # ADPretrain main.py에서 WITH_AUG=True
        if with_aug:
            self.transform = MoCoTransform(
                kwargs.get("crp_size", 224)
            )

        # ----------------------------
        # Mask transform
        # ----------------------------
        self.target_transform = T.Compose([
            T.Resize(
                kwargs.get("msk_size", 224),
                T.InterpolationMode.NEAREST,
            ),

            T.CenterCrop(
                kwargs.get("msk_crp_size", 224)
            ),

            T.ToTensor(),
        ])

        # ----------------------------
        # Load dataset
        # ----------------------------
        self.data_items_by_view = {
            "view1": []
        }

        self._load_data()

        self.data_items_by_class = {
            "weld": self.data_items_by_view
        }

        # reference normal images
        normal_paths = [
            item["image_path"]
            for item
            in self.data_items_by_view["view1"]
            if item["label"] == 0
        ]

        self.normal_images = {
            "weld": {
                "view1": normal_paths
            }
        }

        self.num_images_per_view = {
            "view1": len(
                self.data_items_by_view["view1"]
            )
        }

        self.class_to_idx = {
            "weld": 0
        }

        self.idx_to_class = {
            0: "weld"
        }

        self._print_summary()

    # ==========================================================
    # Utils
    # ==========================================================

    @staticmethod
    def _find_images(directory):

        directory = Path(directory)

        image_paths = []

        for p in directory.iterdir():

            if not p.is_file():
                continue

            if p.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png",
            ]:
                image_paths.append(p)

        return sorted(image_paths)

    # ==========================================================
    # Load data
    # ==========================================================

    def _load_data(self):

        # -------------------------------------
        # Normal
        # -------------------------------------

        normal_dir = (
            self.images_root /
            "normal"
        )

        if not normal_dir.exists():
            raise FileNotFoundError(
                f"Normal directory not found: "
                f"{normal_dir}"
            )

        normal_images = self._find_images(
            normal_dir
        )

        for image_path in normal_images:

            data_item = {
                "image_path": str(image_path),
                "label": 0,
                "mask_path": None,
                "anomaly_type": "good",
                "class_name": "weld",
            }

            self.data_items_by_view[
                "view1"
            ].append(data_item)

        # -------------------------------------
        # Defects
        # -------------------------------------

        missing_masks = []

        for defect_type in self.DEFECT_TYPES:

            image_dir = (
                self.images_root /
                defect_type
            )

            mask_dir = (
                self.masks_root /
                defect_type
            )

            if not image_dir.exists():

                raise FileNotFoundError(
                    f"Image directory "
                    f"not found: {image_dir}"
                )

            if not mask_dir.exists():

                raise FileNotFoundError(
                    f"Mask directory "
                    f"not found: {mask_dir}"
                )

            image_paths = self._find_images(
                image_dir
            )

            for image_path in image_paths:

                mask_path = (
                    mask_dir /
                    f"{image_path.stem}_mask.png"
                )

                if not mask_path.exists():

                    missing_masks.append(
                        str(mask_path)
                    )

                    continue

                data_item = {
                    "image_path":
                        str(image_path),

                    "label":
                        1,

                    "mask_path":
                        str(mask_path),

                    "anomaly_type":
                        defect_type,

                    "class_name":
                        "weld",
                }

                self.data_items_by_view[
                    "view1"
                ].append(data_item)

        if missing_masks:

            print(
                f"[WARNING] "
                f"Missing masks: "
                f"{len(missing_masks)}"
            )

            for p in missing_masks[:10]:
                print("   ", p)

            raise RuntimeError(
                "Some anomaly images "
                "do not have masks."
            )

    # ==========================================================
    # Dataset API
    # ==========================================================

    def __len__(self):

        return len(
            self.data_items_by_view[
                "view1"
            ]
        )

    def __getitem__(self, idx):

        data_item = (
            self.data_items_by_view[
                "view1"
            ][idx]
        )

        image_path = (
            data_item["image_path"]
        )

        label = (
            data_item["label"]
        )

        mask_path = (
            data_item["mask_path"]
        )

        class_name = "weld"
        image_view = "view1"

        image, mask = (
            self._load_image_and_mask(
                image_path,
                label,
                mask_path,
            )
        )

        return (
            image,
            label,
            mask,
            class_name,
            image_view,
        )

    # ==========================================================
    # Load image / mask
    # ==========================================================

    def _load_image_and_mask(
        self,
        image_path,
        label,
        mask_path,
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(
            image
        )

        # normal
        if label == 0:

            mask = torch.zeros(
                [
                    1,
                    self.cropsize[0],
                    self.cropsize[1],
                ]
            )

        # anomaly
        else:

            mask = Image.open(
                mask_path
            ).convert("L")

            mask = np.array(
                mask
            )

            # binary mask
            mask[
                mask != 0
            ] = 255

            mask = Image.fromarray(
                mask
            )

            mask = (
                self.target_transform(
                    mask
                )
            )

        return image, mask

    # ==========================================================
    # ADPretrain normal reference
    # ==========================================================

    def get_random_normal_images(
        self,
        class_name,
        view,
        num_shot=4,
    ):

        if class_name != "weld":

            raise ValueError(
                f"Unknown class: "
                f"{class_name}"
            )

        if view != "view1":

            raise ValueError(
                f"Unknown view: "
                f"{view}"
            )

        normal_images = (
            self.normal_images[
                "weld"
            ][
                "view1"
            ]
        )

        if len(normal_images) == 0:

            raise RuntimeError(
                "No normal images found."
            )

        # RealIAD와 동일하게
        # replacement sampling
        n_idxs = np.random.randint(
            len(normal_images),
            size=num_shot,
        )

        return [
            normal_images[i]
            for i in n_idxs
        ]

    # ==========================================================
    # Summary
    # ==========================================================

    def _print_summary(self):

        items = (
            self.data_items_by_view[
                "view1"
            ]
        )

        normal = sum(
            x["label"] == 0
            for x in items
        )

        anomaly = sum(
            x["label"] == 1
            for x in items
        )

        print(
            "\n========== WeldAD =========="
        )

        print(
            f"Root      : {self.root}"
        )

        print(
            f"Total     : {len(items)}"
        )

        print(
            f"Normal    : {normal}"
        )

        print(
            f"Anomaly   : {anomaly}"
        )

        for defect_type in self.DEFECT_TYPES:

            n = sum(
                x[
                    "anomaly_type"
                ] == defect_type
                for x in items
            )

            print(
                f"{defect_type:24s}: {n}"
            )

        print(
            "============================\n"
        )