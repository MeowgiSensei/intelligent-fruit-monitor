from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m-seg.pt')

    model.train(
        data='fruitseg_yolo/data.yaml',
        epochs=50,
        imgsz=640,
        batch=32,
        device='0',
        lr0=0.001,
        augment=True,
        # ── Geometric ──────────────────
        degrees=45,            # rotation
        fliplr=0.5,            # flip left/right
        flipud=0.2,            # flip up/down
        translate=0.2,         # shift image
        scale=0.5,             # zoom in/out
        shear=10.0,            # shear angle
        perspective=0.0005,    # perspective warp
        # ── Augment ──────────────────
        mosaic=1.0,            # combine 4 images
        copy_paste=0.5,        # paste objects
        mixup=0.2,             # blend 2 images
        erasing=0.4,           # random erasing
        # ── Colour ─────────────────────
        hsv_h=0.015,           # hue shift
        hsv_s=0.7,             # saturation
        hsv_v=0.4,             # brightness
        # ── NAME ──────────────────   
        name='fruitseg_new_v005'
    )

    print("Training complete!")