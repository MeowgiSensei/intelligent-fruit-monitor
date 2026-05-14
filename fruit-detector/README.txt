🍎 INTELLIGENT FRUIT MONITOR
==============================
Project:    ICT Capstone 9785
Group:      2026-S1-09
University: University of Canberra
 
=======================================
TRAINED MODEL DOWNLOAD
=======================================
 
The trained model (best.pt) is not included in this repo
due to file size. Download it from Google Drive:
 
https://drive.google.com/drive/folders/1czqGULejfeLOy8Cje1_CW2FSJqvIYgQr?usp=sharing
 
After downloading, place the file at:
runs/segment/fruitseg_new_v005/weights/best.pt

Create the folders if they don't exist:
mkdir runs\segment\fruitseg_new_v005\weights\
 
 
=======================================
WHAT YOU NEED
=======================================
 
- Python 3.10 or higher
- NVIDIA GPU with CUDA (recommended)
- dotnet SDK (for the dashboard)
- ~10GB free disk space
 
 
=======================================
STEP 1 — INSTALL REQUIREMENTS
=======================================
 
pip install ultralytics flask flask-cors opencv-python numpy scikit-learn segment-anything torch torchvision
 
 
=======================================
STEP 2 — DOWNLOAD SAM WEIGHTS
=======================================
 
Only needed if you want to add new fruits and generate masks.
Run this once:
 
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'sam_vit_h.pth'
)
print('Done!')
"
 
 
=======================================
HOW TO RUN THE SYSTEM
=======================================
 
Every time you use the system open TWO terminals:
 
TERMINAL 1 — Start the AI (run this first):
   cd fruit-detector
   python api.py
 
   You should see:
   Fruit Detection API starting...
   Running on http://localhost:5002
 
   Keep this window open the whole time!
 
TERMINAL 2 — Start the Dashboard:
   cd FruitVision_Web
   dotnet run
 
   Wait until you see:
   Now listening on: http://localhost:5000
 
OPEN BROWSER:
   http://localhost:5000/Demo
 
 
DETECTION PAGES
---------------
Photo Detection  -> Photo tab
Video Detection  -> Video tab
Live Camera      -> Live Camera tab
 
 
=======================================
HOW TO TRAIN THE MODEL
=======================================
 
FIRST TIME TRAINING (using FruitSeg30 dataset)
-----------------------------------------------
1. Make sure FruitSeg30/ folder is in fruit-detector/
 
2. Prepare the dataset:
   python prepare_fruitseg.py
 
3. Train the model:
   python train.py
 
4. Model saves to:
   runs/segment/[name]/weights/best.pt
 
5. Update api.py to point to new model:
   model = YOLO('runs/segment/[name]/weights/best.pt')
 
 
TO ADD NEW FRUITS AND RETRAIN
------------------------------
1. Create a new folder inside FruitSeg30/:
   FruitSeg30/[FruitName]/Images/
   FruitSeg30/[FruitName]/Mask/     <- SAM fills this automatically
 
2. Drop fruit photos into:
   FruitSeg30/[FruitName]/Images/
   Name photos as: add_001.jpg, add_002.jpg etc
 
3. Run SAM to generate masks and rebuild dataset:
   python add_fruit.py
 
4. Train the model:
   python train.py
 
5. Update api.py with new model path
 
 
NAMING RULES FOR NEW FRUIT FOLDERS
------------------------------------
Good:  Pineapple
Good:  Apple_Red
Good:  Dragon_Fruit
Bad:   Apple Red     (no spaces allowed)
Bad:   pineapple     (must capitalise first letter)
 
 
=======================================
FILE STRUCTURE
=======================================
 
fruit-detector/
├── api.py                -> AI detection API (run this!)
├── train.py              -> trains the model
├── prepare_fruitseg.py   -> prepares FruitSeg30 dataset
├── add_fruit.py          -> adds new fruits + generates SAM masks
├── FruitSeg30/           -> training dataset (20 fruit classes)
├── fruitseg_yolo/        -> prepared YOLO dataset (auto generated)
├── runs/                 -> trained models (download best.pt separately)
└── yolov8m-seg.pt        -> base model for retraining
 
 
=======================================
MODEL INFORMATION
=======================================
 
Model:      YOLOv8m-seg
Dataset:    FruitSeg30 (30 fruit classes, ~2,000 images)
Accuracy:   99.4% mAP50 detection
            98.3% mask mAP segmentation
Training:   50 epochs, GPU accelerated (RTX 5070 Ti)
Speed:      ~28ms per image inference
 
 
=======================================
COMMON PROBLEMS
=======================================
 
API Offline in dashboard   -> Make sure Terminal 1 is running python api.py
Website not loading        -> Make sure Terminal 2 is running dotnet run
pip not recognized         -> Reinstall Python and tick Add Python to PATH
Camera not working         -> Click Allow when browser asks for camera permission
Model not found error      -> Download best.pt from Google Drive link above