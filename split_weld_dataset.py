# split_weld_dataset.py

from pathlib import Path
import random
import shutil


# ============================================================
# Configuration
# ============================================================

SOURCE_DIR = Path("./dataset_processed")

# Main welding dataset
MAIN_DATASET_DIR = Path("./data/weld")

# ADPretrain 8-shot reference dataset
REFERENCE_DATASET_DIR = Path("./data/8shot/weld")

DATASET_NAME = "weld"

# 전체 정상 이미지 중 test에 사용할 비율
TEST_RATIO = 0.2

# ADPretrain reference shot 수
REFERENCE_COUNT = 8

# 동일한 실험 재현을 위한 seed
SEED = 42

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# Destination directories
# ============================================================

train_dir = (
    MAIN_DATASET_DIR
    / DATASET_NAME
    / "train"
    / "good"
)

test_dir = (
    MAIN_DATASET_DIR
    / DATASET_NAME
    / "test"
    / "good"
)

reference_dir = (
    REFERENCE_DATASET_DIR
    / DATASET_NAME
    / "train"
    / "good"
)


# ============================================================
# Read images
# ============================================================

image_paths = sorted([
    p
    for p in SOURCE_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in SUPPORTED_EXTENSIONS
])

num_total = len(image_paths)

if num_total == 0:
    raise RuntimeError(
        f"No images found in {SOURCE_DIR}"
    )

print(f"Total normal images: {num_total}")


# ============================================================
# Shuffle reproducibly
# ============================================================

random.seed(SEED)
random.shuffle(image_paths)


# ============================================================
# Train / Test split
# ============================================================

num_test = max(
    1,
    round(num_total * TEST_RATIO)
)

test_images = image_paths[:num_test]
train_images = image_paths[num_test:]


if len(train_images) < REFERENCE_COUNT:
    raise RuntimeError(
        f"Train set has only {len(train_images)} images, "
        f"but {REFERENCE_COUNT} reference images are required."
    )


# ============================================================
# Select references FROM train set
#
# Important:
# reference images are COPIED from training normal images.
# They are NOT removed from the train set.
# ============================================================

reference_images = random.sample(
    train_images,
    REFERENCE_COUNT
)


# ============================================================
# Print split information
# ============================================================

print()
print("========== Split ==========")
print(f"Total      : {num_total}")
print(f"Train good : {len(train_images)}")
print(f"Test good  : {len(test_images)}")
print(f"Reference  : {len(reference_images)}")
print("===========================")
print()

print(
    "Reference images are selected from "
    "the training normal set."
)


# ============================================================
# Create directories
# ============================================================

for directory in [
    train_dir,
    test_dir,
    reference_dir,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Safety check
#
# 기존 split 결과가 있으면 그대로 덮어쓰지 않도록
# 필요하면 아래 clear_directory()를 사용
# ============================================================

def clear_directory(directory: Path):
    for path in directory.iterdir():

        if path.is_file():
            path.unlink()

        elif path.is_dir():
            shutil.rmtree(path)


clear_directory(train_dir)
clear_directory(test_dir)
clear_directory(reference_dir)


# ============================================================
# Copy helper
# ============================================================

def copy_images(images, destination):

    for index, src in enumerate(images):

        # 원래 파일 이름을 보존하면서 순번 추가
        dst_name = f"{index:04d}_{src.name}"

        dst = destination / dst_name

        shutil.copy2(
            src,
            dst
        )


# ============================================================
# Copy
# ============================================================

copy_images(
    train_images,
    train_dir
)

copy_images(
    test_images,
    test_dir
)

copy_images(
    reference_images,
    reference_dir
)


# ============================================================
# Print reference filenames
# ============================================================

print()
print("8-shot reference images:")

for path in reference_images:
    print(f"  - {path.name}")


print()
print("Dataset preparation completed.")
print()
print(f"Train     : {train_dir}")
print(f"Test good : {test_dir}")
print(f"Reference : {reference_dir}")