# prepare_weld_defects.py

from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageOps


# ============================================================
# Configuration
# ============================================================

# CVAT에서 export한 annotations.xml + 원본 defect images
SOURCE_DIR = Path("./dataset/defect_dataset")

ANNOTATION_FILE = SOURCE_DIR / "annotations.xml"

# split_weld_dataset.py가 만든 main dataset
MAIN_DATASET_DIR = Path("./data/weld")

DATASET_NAME = "weld"

# CVAT에서 defect polygon에 사용한 label
DEFECT_LABEL = "defect"


# ============================================================
# Destination directories
# ============================================================

defect_dir = (
    MAIN_DATASET_DIR
    / DATASET_NAME
    / "test"
    / "defect"
)

ground_truth_dir = (
    MAIN_DATASET_DIR
    / DATASET_NAME
    / "ground_truth"
    / "defect"
)


# ============================================================
# Helper: find source image
# ============================================================

def find_image(source_dir: Path, image_name: str):
    """
    annotations.xml의 image name과 동일한 파일을
    SOURCE_DIR 아래에서 검색합니다.
    """

    exact_path = source_dir / image_name

    if exact_path.exists():
        return exact_path

    # 하위 폴더에 들어있는 경우까지 검색
    matches = list(source_dir.rglob(image_name))

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        print(
            f"[WARNING] Multiple files found for {image_name}. "
            f"Using: {matches[0]}"
        )

    return matches[0]


# ============================================================
# Helper: parse CVAT polygon points
# ============================================================

def parse_polygon_points(points_string: str):
    """
    CVAT polygon format:

    "x1,y1;x2,y2;x3,y3;..."

    -> [(x1, y1), (x2, y2), ...]
    """

    points = []

    for pair in points_string.split(";"):

        x, y = pair.split(",")

        points.append(
            (
                float(x),
                float(y),
            )
        )

    return points


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check annotation file
    # --------------------------------------------------------

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"annotations.xml not found:\n"
            f"{ANNOTATION_FILE}"
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    defect_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ground_truth_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Parse CVAT XML
    # --------------------------------------------------------

    tree = ET.parse(ANNOTATION_FILE)
    root = tree.getroot()

    image_elements = root.findall(".//image")

    print()
    print("========================================")
    print(" CVAT Weld Defect Dataset Preparation")
    print("========================================")
    print(f"Annotation file : {ANNOTATION_FILE}")
    print(f"Images in XML   : {len(image_elements)}")
    print()

    processed_count = 0
    polygon_count = 0
    missing_images = []

    # --------------------------------------------------------
    # Process each annotated image
    # --------------------------------------------------------

    for image_element in image_elements:

        image_name = image_element.attrib["name"]

        xml_width = int(image_element.attrib["width"])
        xml_height = int(image_element.attrib["height"])

        print(f"[IMAGE] {image_name}")

        # ----------------------------------------------------
        # Find source image
        # ----------------------------------------------------

        source_image_path = find_image(
            SOURCE_DIR,
            image_name,
        )

        if source_image_path is None:

            print(
                f"  [ERROR] Original image not found."
            )

            missing_images.append(image_name)

            continue

        # ----------------------------------------------------
        # Open original image
        # ----------------------------------------------------

        image = Image.open(source_image_path)
        raw_image = Image.open(source_image_path)

        print(f"  XML size       : {xml_width} x {xml_height}")
        print(f"  PIL raw size   : {raw_image.size}")
        print(f"  EXIF orientation: {raw_image.getexif().get(274)}")

        image = ImageOps.exif_transpose(raw_image)

        print(f"  Transposed size: {image.size}")

        image = image.convert("RGB")

        # JPEG EXIF orientation을 실제 pixel 방향에 적용
        image = ImageOps.exif_transpose(image)

        image = image.convert("RGB")

        image_width, image_height = image.size

        # XML image size와 실제 image size 확인
        if (
            image_width != xml_width
            or image_height != xml_height
        ):
            print(
                "  [WARNING] Image size mismatch:"
            )
            print(
                f"    XML    : {xml_width} x {xml_height}"
            )
            print(
                f"    Actual : {image_width} x {image_height}"
            )

        # ----------------------------------------------------
        # Output image name
        #
        # image11.jpeg
        #       ↓
        # image11.png
        # ----------------------------------------------------

        stem = Path(image_name).stem

        output_image_path = (
            defect_dir
            / f"{stem}.png"
        )

        output_mask_path = (
            ground_truth_dir
            / f"{stem}_mask.png"
        )

        # ----------------------------------------------------
        # Save defect image as PNG
        # ----------------------------------------------------

        image.save(
            output_image_path,
            format="PNG",
        )

        # ----------------------------------------------------
        # Create binary mask
        #
        # background = 0
        # defect     = 255
        # ----------------------------------------------------

        mask = Image.new(
            mode="L",
            size=(image_width, image_height),
            color=0,
        )

        draw = ImageDraw.Draw(mask)

        current_polygon_count = 0

        # ----------------------------------------------------
        # Read polygons
        # ----------------------------------------------------

        polygons = image_element.findall("polygon")

        for polygon in polygons:

            label = polygon.attrib.get(
                "label",
                "",
            )

            # defect annotation만 사용
            if label != DEFECT_LABEL:
                continue

            points_string = polygon.attrib.get(
                "points"
            )

            if not points_string:
                continue

            points = parse_polygon_points(
                points_string
            )

            # CVAT annotation 좌표계 → 현재 실제 image 좌표계
            scale_x = image_width / xml_width
            scale_y = image_height / xml_height

            points = [
                (
                    x * scale_x,
                    y * scale_y,
                )
                for x, y in points
            ]

            if len(points) < 3:
                print(
                    "  [WARNING] "
                    "Polygon has fewer than 3 points."
                )
                continue

            # polygon 내부를 white(255)로 채움
            draw.polygon(
                points,
                fill=255,
            )

            current_polygon_count += 1
            polygon_count += 1

        # ----------------------------------------------------
        # Save mask
        # ----------------------------------------------------

        mask.save(
            output_mask_path,
            format="PNG",
        )

        processed_count += 1

        print(
            f"  Original : {source_image_path}"
        )

        print(
            f"  Output   : {output_image_path}"
        )

        print(
            f"  Mask     : {output_mask_path}"
        )

        print(
            f"  Polygons : {current_polygon_count}"
        )

        if current_polygon_count == 0:
            print(
                "  [WARNING] No defect polygon found."
            )

        print()

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("========================================")
    print(" Completed")
    print("========================================")

    print(
        f"Processed images : {processed_count}"
    )

    print(
        f"Defect polygons  : {polygon_count}"
    )

    print(
        f"Missing images   : {len(missing_images)}"
    )

    print()

    print(
        f"Defect images:\n"
        f"  {defect_dir}"
    )

    print()

    print(
        f"Ground-truth masks:\n"
        f"  {ground_truth_dir}"
    )

    if missing_images:

        print()
        print("Missing original images:")

        for image_name in missing_images:
            print(
                f"  - {image_name}"
            )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()