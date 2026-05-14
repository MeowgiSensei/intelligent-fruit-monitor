import os
import cv2
import shutil
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# ── CONFIG — only change these ──────────────────
FRUITSEG_DIR = 'FruitSeg30'
OUTPUT_DIR   = 'fruitseg_yolo'
TRAIN_SPLIT  = 0.8
SAM_CHECKPOINT = 'sam_vit_h.pth'
MODEL_TYPE     = 'vit_h'
# ────────────────────────────────────────────────

def get_best_mask(masks, img_h, img_w):
    if not masks:
        return None

    cx, cy = img_w // 2, img_h // 2
    best_mask  = None
    best_score = -1
    total      = img_h * img_w

    for m in masks:
        seg  = m['segmentation']
        area = m['area']

        # Skip masks that are too small or too large
        if area < total * 0.05 or area > total * 0.70:
            continue

        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue

        # Skip masks that touch the bottom edge (ground/floor)
        if ys.max() >= img_h * 0.95:
            continue

        # Score based on centrality
        dist  = ((xs.mean()-cx)**2 + (ys.mean()-cy)**2)**0.5
        score = area - dist * 10

        if score > best_score:
            best_score = score
            best_mask  = seg

    return best_mask

def run_sam(fruit_name):
    """Generate SAM masks for images that don't have masks yet"""
    images_dir = os.path.join(FRUITSEG_DIR, fruit_name, 'Images')
    masks_dir  = os.path.join(FRUITSEG_DIR, fruit_name, 'Mask')
    os.makedirs(masks_dir, exist_ok=True)

    # Find images without masks
    existing_masks = set(os.listdir(masks_dir))
    images = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg','.jpeg','.png'))
        and os.path.splitext(f)[0] + '_mask.png' not in existing_masks
    ]

    if not images:
        print(f"  ✅ All masks already exist for {fruit_name}")
        return

    print(f"  Generating SAM masks for {len(images)} images...")

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam    = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=500,
    )

    generated = 0
    for img_file in images:
        img_path = os.path.join(images_dir, img_file)
        image    = cv2.imread(img_path)
        if image is None:
            continue
        img_h, img_w = image.shape[:2]
        image_rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks        = mask_gen.generate(image_rgb)
        best         = get_best_mask(masks, img_h, img_w)
        if best is None:
            continue
        base      = os.path.splitext(img_file)[0]
        mask_path = os.path.join(masks_dir, base + '_mask.png')
        cv2.imwrite(mask_path, (best * 255).astype(np.uint8))
        generated += 1

    print(f"  ✅ Generated {generated} masks for {fruit_name}")

def mask_to_yolo(mask_path, class_idx, img_w, img_h):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = cv2.resize(mask, (img_w, img_h))
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return None
    points = contour.reshape(-1, 2)
    norm   = [f"{max(0.0,min(1.0,x/img_w)):.6f} {max(0.0,min(1.0,y/img_h)):.6f}" for x,y in points]
    return f"{class_idx} {' '.join(norm)}"

def prepare_dataset():
    """Convert FruitSeg30 to YOLO format"""
    fruit_classes = sorted([
        f for f in os.listdir(FRUITSEG_DIR)
        if os.path.isdir(os.path.join(FRUITSEG_DIR, f))
    ])

    print(f"\nFound {len(fruit_classes)} fruit classes:")
    for i, f in enumerate(fruit_classes):
        print(f"  {i}: {f}")

    # Clear and recreate output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ['train', 'val']:
        os.makedirs(f'{OUTPUT_DIR}/{split}/images', exist_ok=True)
        os.makedirs(f'{OUTPUT_DIR}/{split}/labels', exist_ok=True)

    total_train = 0
    total_val   = 0
    skipped     = 0

    for class_idx, fruit_name in enumerate(fruit_classes):
        images_dir = os.path.join(FRUITSEG_DIR, fruit_name, 'Images')
        masks_dir  = os.path.join(FRUITSEG_DIR, fruit_name, 'Mask')

        if not os.path.exists(images_dir):
            continue

        # Build mask lookup
        mask_lookup = {}
        if os.path.exists(masks_dir):
            for f in os.listdir(masks_dir):
                if '_mask' in f:
                    base = f.replace('_mask.png','').replace('_mask.jpg','')
                    mask_lookup[base] = os.path.join(masks_dir, f)

        # Find valid pairs
        valid_pairs = []
        for f in os.listdir(images_dir):
            if not f.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            base = os.path.splitext(f)[0]
            if base in mask_lookup:
                valid_pairs.append((
                    os.path.join(images_dir, f),
                    mask_lookup[base],
                    f
                ))
            else:
                skipped += 1

        if not valid_pairs:
            print(f"⚠️  {fruit_name} — no valid pairs")
            continue

        train_pairs, val_pairs = train_test_split(
            valid_pairs, test_size=1-TRAIN_SPLIT, random_state=42)

        for split, pairs in [('train', train_pairs), ('val', val_pairs)]:
            for img_path, mask_path, img_file in pairs:
                img = cv2.imread(img_path)
                if img is None:
                    skipped += 1
                    continue
                h, w  = img.shape[:2]
                label = mask_to_yolo(mask_path, class_idx, w, h)
                if label is None:
                    skipped += 1
                    continue
                base = os.path.splitext(img_file)[0]
                shutil.copy2(img_path,
                    f'{OUTPUT_DIR}/{split}/images/{fruit_name}_{base}.jpg')
                with open(f'{OUTPUT_DIR}/{split}/labels/{fruit_name}_{base}.txt','w') as lf:
                    lf.write(label + '\n')
                if split == 'train': total_train += 1
                else: total_val += 1

        print(f"✅ {fruit_name}: {len(train_pairs)} train, {len(val_pairs)} val")

    # Save data.yaml
    abs_out = os.path.abspath(OUTPUT_DIR).replace('\\','/')
    with open(f'{OUTPUT_DIR}/data.yaml','w') as yf:
        yf.write(f"train: {abs_out}/train/images\n")
        yf.write(f"val: {abs_out}/val/images\n\n")
        yf.write(f"nc: {len(fruit_classes)}\n")
        yf.write(f"names: {fruit_classes}\n")

    print(f"\n{'='*40}")
    print(f"✅ Training images:   {total_train}")
    print(f"✅ Validation images: {total_val}")
    print(f"⚠️  Skipped:          {skipped}")

if __name__ == '__main__':
    # Step 1 — Generate SAM masks for any fruit missing masks
    print("=== Step 1: Generating SAM masks ===")
    for fruit in sorted(os.listdir(FRUITSEG_DIR)):
        fruit_path = os.path.join(FRUITSEG_DIR, fruit)
        if not os.path.isdir(fruit_path):
            continue
        imgs_dir = os.path.join(fruit_path, 'Images')
        if not os.path.exists(imgs_dir):
            continue
        needs_mask = [
            f for f in os.listdir(imgs_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
            and not os.path.exists(os.path.join(
                fruit_path, 'Mask',
                os.path.splitext(f)[0] + '_mask.png'))
        ]
        if needs_mask:
            print(f"\nProcessing {fruit} ({len(needs_mask)} images need masks)...")
            run_sam(fruit)
        else:
            print(f"✅ {fruit} — masks already exist")

    # Step 2 — Prepare dataset
    print("\n=== Step 2: Preparing YOLO dataset ===")
    prepare_dataset()

    print("\n=== Done! Now run train.py ===")