#!/usr/bin/env python3
"""Weld inference: aligned DINO input grid and patch occlusion sensitivity.

Run from the original project root (the project modules/checkpoint are required).
The pixel map is pooled only for candidate ranking; it is NOT the raw PatchCore
patch-distance grid. Occlusion changes pixels and reruns the full fitted model.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224


def input_geometry(width, height):
    """Match torchvision Resize(224) + CenterCrop(224) for PIL input."""
    if width <= height:
        rw, rh = INPUT_SIZE, int(INPUT_SIZE * height / width)
    else:
        rw, rh = int(INPUT_SIZE * width / height), INPUT_SIZE
    left = int(round((rw - INPUT_SIZE) / 2.0))
    top = int(round((rh - INPUT_SIZE) / 2.0))
    return rw, rh, left, top


def original_box(box, original_size):
    width, height = original_size
    rw, rh, left, top = input_geometry(width, height)
    x0, y0, x1, y1 = box
    return [(x0 + left) * width / rw, (y0 + top) * height / rh,
            (x1 + left) * width / rw, (y1 + top) * height / rh]


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def pixel_map_224(value):
    value = np.squeeze(as_numpy(value)).astype(np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"Expected finite 2D pixel anomaly map; got {value.shape}")
    if value.shape != (INPUT_SIZE, INPUT_SIZE):
        value = np.asarray(Image.fromarray(value).resize(
            (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR))
    return value


def patch_records(pixel_map, patch_size, original_size):
    if patch_size <= 0 or INPUT_SIZE % patch_size:
        raise ValueError("patch_size must divide 224 exactly")
    n = INPUT_SIZE // patch_size
    records = []
    for row in range(n):
        for col in range(n):
            x0, y0 = col * patch_size, row * patch_size
            box = [x0, y0, x0 + patch_size, y0 + patch_size]
            region = pixel_map[y0:y0 + patch_size, x0:x0 + patch_size]
            obox = original_box(box, original_size)
            records.append({
                "patch_id": row * n + col, "row": row, "col": col,
                "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                "original_x0": obox[0], "original_y0": obox[1],
                "original_x1": obox[2], "original_y1": obox[3],
                "pooled_map_mean": float(region.mean()),
                "pooled_map_max": float(region.max()),
                "map_rank": None, "occluded_score": None,
                "delta_score": None, "occluded_prediction": None,
                "label_flipped": None,
            })
    # Stable, deterministic ordering for equal scores.
    ranked = sorted(records, key=lambda r: (-r["pooled_map_mean"], r["patch_id"]))
    for rank, record in enumerate(ranked, 1):
        record["map_rank"] = rank
    return records, ranked


def draw_box(ax, box, color="lime", label=None, linewidth=1.8):
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                           fill=False, edgecolor=color, linewidth=linewidth))
    if label is not None:
        ax.text(x0, y0, str(label), color="white", fontsize=7, va="top",
                bbox=dict(facecolor="black", alpha=0.75, pad=1, edgecolor="none"))


def visualize(original, model_rgb, pixel_map, records, ranked, score, threshold,
              prediction, patch_size, topk, output):
    shown = ranked[:min(topk, len(ranked))]
    n = INPUT_SIZE // patch_size
    pooled = np.array([r["pooled_map_mean"] for r in records]).reshape(n, n)
    # These bounds control colors only. Never use them to classify the image.
    vmin, vmax = float(pixel_map.min()), float(pixel_map.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    fig, axs = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    axs[0, 0].imshow(original)
    crop_box = original_box([0, 0, INPUT_SIZE, INPUT_SIZE], original.size)
    draw_box(axs[0, 0], crop_box, "cyan", "Model crop")
    for r in shown:
        draw_box(axs[0, 0], [r[k] for k in
                 ("original_x0", "original_y0", "original_x1", "original_y1")],
                 label=f"P{r['patch_id']}")
    axs[0, 0].set_title("Original: crop and mapped candidate regions")

    axs[0, 1].imshow(model_rgb)
    for boundary in range(0, INPUT_SIZE + 1, patch_size):
        axs[0, 1].axvline(boundary - 0.5, color="white", alpha=0.5, lw=0.5)
        axs[0, 1].axhline(boundary - 0.5, color="white", alpha=0.5, lw=0.5)
    for r in shown:
        draw_box(axs[0, 1], [r[k] for k in ("x0", "y0", "x1", "y1")],
                 label=f"P{r['patch_id']}")
    axs[0, 1].set_title(f"Actual model input: {n} x {n} patches ({patch_size}px)")

    im = axs[0, 2].imshow(pooled, cmap="inferno", interpolation="nearest",
                           vmin=vmin, vmax=vmax)
    for r in shown:
        axs[0, 2].text(r["col"], r["row"], f"P{r['patch_id']}",
                       ha="center", va="center", fontsize=6, color="cyan")
    axs[0, 2].set_title("Mean pixel anomaly per block (approximate)")
    fig.colorbar(im, ax=axs[0, 2], shrink=0.8)

    im = axs[1, 0].imshow(pixel_map, cmap="inferno", vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("Pixel anomaly map: model coordinates")
    fig.colorbar(im, ax=axs[1, 0], shrink=0.8)
    axs[1, 1].imshow(model_rgb)
    axs[1, 1].imshow(pixel_map, cmap="inferno", vmin=vmin, vmax=vmax, alpha=0.45)
    for r in shown:
        draw_box(axs[1, 1], [r[k] for k in ("x0", "y0", "x1", "y1")],
                 label=f"P{r['patch_id']}")
    axs[1, 1].set_title("Aligned overlay: highest map-ranked blocks")

    tested = [r for r in records if r["delta_score"] is not None]
    if tested:
        tested = sorted(tested, key=lambda r: r["delta_score"], reverse=True)[:topk]
        delta = [r["delta_score"] for r in tested]
        labels = [f"P{r['patch_id']}" + (" *" if r["label_flipped"] else "")
                  for r in tested]
        axs[1, 2].barh(labels, delta,
                       color=["tomato" if d > 0 else "steelblue" for d in delta])
        axs[1, 2].invert_yaxis()
        axs[1, 2].axvline(0, color="black", lw=0.8)
        axs[1, 2].set_xlabel("Original score - masked-patch score")
        axs[1, 2].set_title("Occlusion sensitivity (* = label changed)")
    else:
        axs[1, 2].text(0.5, 0.5, "Occlusion disabled\nMap ranks indicate localization only",
                       ha="center", va="center", transform=axs[1, 2].transAxes)
        axs[1, 2].axis("off")
    for ax in axs.flat[:5]:
        ax.axis("off")
    fig.suptitle(f"{prediction} | score={score:.6f} | threshold={threshold:.6f} | "
                 f"gap={score - threshold:+.6f}\n"
                 "Patch IDs are zero-based, row-major. Map colors are relative; "
                 "occlusion is sensitivity, not defect ground truth.", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def tensor_loader(tensor):
    import torch
    from torch.utils.data import Dataset, DataLoader

    class OneTensor(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return tensor, torch.tensor(0, dtype=torch.long), torch.zeros(
                1, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32), "weld"

    return DataLoader(OneTensor(), batch_size=1, shuffle=False, num_workers=0)


def infer(patchcore, tensor, refs):
    import torch
    with torch.no_grad():
        result = patchcore.predict(tensor_loader(tensor), refs, "weld", aligned=False)
    score = float(as_numpy(result[0]).reshape(-1)[0])
    if not np.isfinite(score):
        raise ValueError("Model returned a non-finite image score")
    return score, pixel_map_224(as_numpy(result[1])[0])


def main(args):
    # Allow pure visualization helpers to be imported without the project/GPU.
    sys.path.insert(0, str(Path.cwd()))
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset
    from datasets.mvtec import MVTEC
    from models.dino import DinoModel
    from models.projector import MultiScaleAttentionProjector
    from ad_models.patchcore import get_patchcore
    from main_weld_ddp import DTYPE, NORMALIZE, load_mc_reference_features
    from eval_weld_checkpoint import load_projector_checkpoint

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = DTYPE
    threshold_values = np.asarray(np.load(args.threshold_path)).reshape(-1)
    if threshold_values.size != 1 or not np.isfinite(threshold_values[0]):
        raise ValueError("threshold_path must contain one finite image threshold")
    threshold = float(threshold_values[0])
    original = Image.open(args.image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(INPUT_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(INPUT_SIZE), T.ToTensor(),
        T.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ])
    tensor = transform(original)
    model_rgb = np.clip(as_numpy(tensor).transpose(1, 2, 0) * STD + MEAN, 0, 1)

    encoder = DinoModel(args.backbone, device=str(device))
    encoder.to(dtype).to(device).eval()
    # Try to verify the real token patch size, rather than trusting the default.
    detected = set()
    for name, module in encoder.named_modules():
        if hasattr(module, "patch_size") and ("patch_embed" in name or name == ""):
            size = module.patch_size
            if isinstance(size, int):
                detected.add(size)
            elif isinstance(size, (tuple, list)) and len(size) == 2 and size[0] == size[1]:
                detected.add(int(size[0]))
    if detected and detected != {args.patch_size}:
        raise ValueError(f"Encoder patch sizes {detected} differ from --patch_size {args.patch_size}")
    if not detected:
        print("WARNING: encoder token size not exposed; verify --patch_size manually.")
    if INPUT_SIZE % args.patch_size:
        raise ValueError("patch_size must divide model input size 224")

    projectors = MultiScaleAttentionProjector(encoder.feature_dimensions, device=str(device))
    projectors.to(dtype).to(device)
    checkpoint = load_projector_checkpoint(projectors, args.checkpoint)
    projectors.eval()
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    refs = load_mc_reference_features(args.test_ref_feature_dir, ["weld"],
                                      device, args.num_ref_shot)["weld"]
    normal = MVTEC(args.test_dataset_dir, class_name="weld", train=True,
                   normalize=NORMALIZE, img_size=224, crp_size=224,
                   msk_size=224, msk_crp_size=224)
    if args.max_normal_images > 0 and len(normal) > args.max_normal_images:
        indices = np.random.default_rng(42).choice(len(normal), args.max_normal_images,
                                                  replace=False)
        normal = Subset(normal, indices.tolist())
    if not len(normal):
        raise ValueError("No normal images found for the memory bank")
    normal_loader = DataLoader(normal, batch_size=1, shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=device.type == "cuda", drop_last=False)
    patchcore = get_patchcore(encoder, projectors, device=device, dtype=dtype, residual=True)
    print(f"[1] Fitting memory bank from {len(normal)} normal images...")
    with torch.no_grad():
        patchcore.fit(normal_loader, refs, "weld", aligned=False)
    print("[2] Original image inference...")
    score, amap = infer(patchcore, tensor, refs)
    # ========================================================
    # Raw PatchCore patch scores
    # ========================================================

    raw_patch_scores = patchcore.predict_raw_patch_scores(
        tensor.unsqueeze(0),
        refs,
    )

    raw_patch_scores = raw_patch_scores[0]

    print(
        "Raw patch score shape:",
        raw_patch_scores.shape,
    )
    prediction = "Defect" if score >= threshold else "Normal"
    records, ranked = patch_records(amap, args.patch_size, original.size)
    test_count = min(args.occlusion_topk, len(ranked))
    # All patches in one run use the same replacement. Replacement pixels are
    # either the actual image's per-channel mean or ImageNet mean (normalized 0).
    replacement = (tensor.mean(dim=(1, 2), keepdim=True)
                   if args.mask_fill == "image_mean" else torch.zeros(3, 1, 1))
    print(f"[3] Testing {test_count}/{len(records)} patches by occlusion...")
    for i, r in enumerate(ranked[:test_count], 1):
        masked = tensor.clone()
        masked[:, r["y0"]:r["y1"], r["x0"]:r["x1"]] = replacement
        masked_score, _ = infer(patchcore, masked, refs)
        masked_prediction = "Defect" if masked_score >= threshold else "Normal"
        r.update(occluded_score=masked_score, delta_score=score - masked_score,
                 occluded_prediction=masked_prediction,
                 label_flipped=masked_prediction != prediction)
        print(f"  [{i}/{test_count}] P{r['patch_id']} (row={r['row']}, col={r['col']}): "
              f"score={masked_score:.6f}, delta={r['delta_score']:+.6f}, "
              f"prediction={masked_prediction}, flipped={r['label_flipped']}", flush=True)

    output = Path(args.output)
    visualize(original, model_rgb, amap, records, ranked, score, threshold,
              prediction, args.patch_size, args.topk, output)
    csv_path = output.with_name(output.stem + "_patches.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(ranked)
    report = {
        "image_path": args.image_path, "checkpoint": args.checkpoint,
        "score": score, "threshold": threshold, "prediction": prediction,
        "score_gap": score - threshold, "input_size": INPUT_SIZE,
        "patch_size": args.patch_size, "patch_count": len(records),
        "occlusion_tested": test_count, "mask_fill": args.mask_fill,
        "backbone": args.backbone, "normal_images_used": len(normal),
        "num_ref_shot": args.num_ref_shot,
        "notes": [
            "pooled_map_mean/max summarize the returned pixel map, not raw patch distances.",
            "delta_score = original score minus score with one input patch masked.",
            "Positive delta: the original region raises the anomaly score relative to this mask.",
            "Negative delta: this mask raises the score; it is not proof of normal evidence.",
            "Occlusion may create out-of-distribution inputs and affects contextual features.",
            "Only tested candidates have sensitivity estimates; missing values mean untested.",
            "Scores are not probabilities; the image threshold does not threshold pixel scores.",
            "Threshold must be calibrated with the same checkpoint and scoring/memory-bank setup.",
        ], "patches": ranked,
    }
    report_path = output.with_name(output.stem + "_patches.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nPrediction={prediction}; score={score:.6f}; threshold={threshold:.6f}")
    print(f"Saved: {output}\nSaved: {csv_path}\nSaved: {report_path}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image_path", required=True)
    p.add_argument("--threshold_path", default="./thresholds/patchcore_image_threshold.npy")
    p.add_argument("--test_dataset_dir", default="/ssdg/spl_yejin/aihub_weld/weld_eval_root")
    p.add_argument("--test_ref_feature_dir", default="./ref_features/dinov2-base/weld_8shot")
    p.add_argument("--backbone", default="dinov2-base")
    p.add_argument("--num_ref_shot", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_normal_images", type=int, default=1000)
    p.add_argument("--output", default="./visualizations/external_patches2.png")
    p.add_argument("--patch_size", type=int, default=14,
                   help="DINO token patch size; checked against encoder when exposed")
    p.add_argument("--topk", type=int, default=5, help="Number of candidate boxes/bars to show")
    p.add_argument("--occlusion_topk", type=int, default=5,
                   help="Number of map-ranked blocks to test; 0 disables; 256 tests all at 14px")
    p.add_argument("--mask_fill", choices=["image_mean", "imagenet_mean"], default="image_mean")
    return p


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if args.patch_size <= 0 or INPUT_SIZE % args.patch_size or args.topk < 1 or args.occlusion_topk < 0:
        p.error("patch_size must be positive and divide 224; topk >= 1; occlusion_topk >= 0")
    main(args)