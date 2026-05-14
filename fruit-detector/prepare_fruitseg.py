import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration ──────────────────────────────────────────
SOURCE_DIR = 'FruitSeg30'        # ataset folder
OUTPUT_DIR = 'fruitseg_yolo'     # output folder for YOLO
TRAIN_SPLIT = 0.8                # 80% train, 20% val
# ────────────────────────────────────────────────────────

def mask_to_yolo_polygon(mask_path, class_idx, img_w, img_h):
    """Convert a mask image to YOLO segmentation format"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Threshold mask to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Get largest contour (the fruit)
    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 100:
        return None

    # Normalise points to 0-1
    points = contour.reshape(-1, 2)
    normalised = []
    for x, y in points:
        normalised.append(f"{x/img_w:.6f} {y/img_h:.6f}")

    return f"{class_idx} {' '.join(normalised)}"

def prepare_dataset():
    # Get all fruit class folders
    fruit_classes = sorted([
        f for f in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, f))
    ])

    print(f"Found {len(fruit_classes)} fruit classes:")
    for i, f in enumerate(fruit_classes):
        print(f"  {i}: {f}")

    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(f'{OUTPUT_DIR}/{split}/images', exist_ok=True)
        os.makedirs(f'{OUTPUT_DIR}/{split}/labels', exist_ok=True)

    total_train = 0
    total_val   = 0
    skipped     = 0

    for class_idx, fruit_name in enumerate(fruit_classes):
        images_dir = os.path.join(SOURCE_DIR, fruit_name, 'Images')
        masks_dir  = os.path.join(SOURCE_DIR, fruit_name, 'Mask')

        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"⚠️  Skipping {fruit_name} - missing Images or Mask folder")
            continue

        # Get all images
        images = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not images:
            print(f"⚠️  Skipping {fruit_name} - no images found")
            continue

        # Split into train and val
        train_imgs, val_imgs = train_test_split(
            images, test_size=1-TRAIN_SPLIT, random_state=42
        )

        for split, img_list in [('train', train_imgs), ('val', val_imgs)]:
            for img_file in img_list:
                img_path  = os.path.join(images_dir, img_file)
                base_name = os.path.splitext(img_file)[0]

                # Find matching mask
                mask_file = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    candidate = os.path.join(masks_dir, base_name + '_mask' + ext)
                    if os.path.exists(candidate):
                        mask_file = candidate
                        break

                if mask_file is None:
                    skipped += 1
                    continue

                # Read image dimensions
                img = cv2.imread(img_path)
                if img is None:
                    skipped += 1
                    continue
                h, w = img.shape[:2]

                # Convert mask to YOLO polygon
                yolo_label = mask_to_yolo_polygon(mask_file, class_idx, w, h)
                if yolo_label is None:
                    skipped += 1
                    continue

                # Save image
                out_img = f'{OUTPUT_DIR}/{split}/images/{fruit_name}_{img_file}'
                shutil.copy2(img_path, out_img)

                # Save label
                out_label = f'{OUTPUT_DIR}/{split}/labels/{fruit_name}_{base_name}.txt'
                with open(out_label, 'w') as f:
                    f.write(yolo_label + '\n')

                if split == 'train':
                    total_train += 1
                else:
                    total_val += 1

        print(f"✅ {fruit_name}: {len(train_imgs)} train, {len(val_imgs)} val")

    # Save data.yaml with absolute paths
    abs_path = os.path.abspath(OUTPUT_DIR).replace('\\', '/')
    yaml_content = f"""train: {abs_path}/train/images
    val: {abs_path}/val/images

    nc: {len(fruit_classes)}
    names: {fruit_classes}
    """
    with open(f'{OUTPUT_DIR}/data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\n=== DONE ===")
    print(f"✅ Training images:   {total_train}")
    print(f"✅ Validation images: {total_val}")
    print(f"⚠️  Skipped:          {skipped}")
    print(f"📁 Output folder:    {OUTPUT_DIR}/")
    print(f"📄 data.yaml saved!")

if __name__ == '__main__':
    prepare_dataset()