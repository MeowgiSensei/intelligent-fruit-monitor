from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from collections import Counter
import cv2
import numpy as np
import base64
import os
from datetime import datetime
import json
import csv

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load model once when API starts
model = YOLO('runs/segment/fruitseg_new_v005/weights/best.pt')

def process_image(image):
    """Run detection on image and return results + annotated image"""
    img_h, img_w = image.shape[:2]
    results = model.predict(source=image, conf=0.60, verbose=False)
    result  = results[0]

    detections = []

    if result.masks is not None:
        for box, mask in zip(result.boxes, result.masks):
            label = model.names[int(box.cls[0])]
            conf  = float(box.conf[0])

            # Size estimation
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_width  = int(x2 - x1)
            box_height = int(y2 - y1)

            mask_array   = mask.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_array, (img_w, img_h))
            pixel_area   = int(np.sum(mask_resized > 0.5))
            size_percent = (pixel_area / (img_w * img_h)) * 100

            if size_percent < 5:
                size_label = 'Small'
            elif size_percent < 15:
                size_label = 'Medium'
            else:
                size_label = 'Large'

            detections.append({
                'fruit':        label,
                'confidence':   round(conf * 100, 1),
                'box_width':    box_width,
                'box_height':   box_height,
                'mask_pixels':  pixel_area,
                'size_percent': round(size_percent, 1),
                'size_label':   size_label,
            })

    # ── Generate annotated image with masks drawn
    annotated = result.plot(
        boxes=True,        # draw bounding boxes
        masks=True,        # draw segmentation masks
        labels=True,       # show fruit labels
        conf=True,         # show confidence scores
    )

    # Convert annotated image to base64 to send to dashboard
    _, buffer    = cv2.imencode('.jpg', annotated)
    img_base64   = base64.b64encode(buffer).decode('utf-8')

    # Count per species
    fruit_counts = Counter([d['fruit'] for d in detections])

    return {
        'total_fruits':    len(detections),
        'fruit_counts':    dict(fruit_counts),
        'detections':      detections,
        'annotated_image': img_base64,      # ← image with masks!
        'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# ── Route 1: Health check
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'model': 'YOLOv8m-seg'})

# ── Route 2: Detect from uploaded image
@app.route('/api/detect/photo', methods=['POST'])
def detect_photo():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Read image
        file    = request.files['image']
        img_arr = np.frombuffer(file.read(), np.uint8)
        image   = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Run detection
        result = process_image(image)

        # Save to CSV and JSON
        save_results(result, file.filename)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Route 3: Detect from base64 image
@app.route('/api/detect/frame', methods=['POST'])
def detect_frame():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image
        img_data = base64.b64decode(data['image'])
        img_arr  = np.frombuffer(img_data, np.uint8)
        image    = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Run detection
        result = process_image(image)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Save results to CSV and JSON ─────────────────
def save_results(result, filename='unknown'):
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save JSON
    json_path = f'results/detections_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'filename':   filename,
            'timestamp':  result['timestamp'],
            'total':      result['total_fruits'],
            'counts':     result['fruit_counts'],
            'detections': result['detections']
        }, f, indent=4)

    # Save CSV
    csv_path = f'results/detections_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'fruit', 'confidence', 'box_width',
            'box_height', 'mask_pixels',
            'size_percent', 'size_label'
        ])
        writer.writeheader()
        writer.writerows(result['detections'])

if __name__ == '__main__':
    print("🍎 Fruit Detection API starting...")
    print("📡 Running on http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=False)