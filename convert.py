from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import shutil

# Enable HEIC / HEIF support in Pillow
register_heif_opener()

INPUT_DIR = Path("./dataset/weld_rois_v2")
OUTPUT_DIR = Path("./dataset_processed")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".heif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

success = 0
failed = 0


for src_path in INPUT_DIR.rglob("*"):

    if not src_path.is_file():
        continue

    if src_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[SKIP] Unsupported: {src_path}")
        continue

    # Preserve subfolder structure
    relative_path = src_path.relative_to(INPUT_DIR)

    dst_path = OUTPUT_DIR / relative_path
    dst_path = dst_path.with_suffix(".png")

    dst_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        with Image.open(src_path) as img:

            # Correct phone-camera orientation
            img = ImageOps.exif_transpose(img)

            # Standardize to RGB
            img = img.convert("RGB")

            # Lossless PNG output
            img.save(dst_path, format="PNG")

        print(f"[OK] {src_path} -> {dst_path}")
        success += 1

    except Exception as e:
        print(f"[ERROR] {src_path}")
        print(f"        {e}")
        failed += 1


print()
print("========================")
print(f"Converted : {success}")
print(f"Failed    : {failed}")
print("========================")