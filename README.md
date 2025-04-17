Vehicle Detection Using YOLOv10
Overview
This repository contains my final year project on vehicle detection using the YOLOv10 algorithm on the VEDAI dataset. The system detects various vehicle types (e.g., car, truck, van) in aerial images, with applications in traffic monitoring and surveillance. The project achieves a mean Average Precision (mAP50) of 0.995 on the validation set.
Features

Real-time vehicle detection using YOLOv10s.
Supports 14 vehicle classes from the VEDAI dataset.
Data preprocessing from COCO to YOLO format.
Visualizes detection results on images.

Tech Stack

Programming Language: Python
Libraries: Ultralytics, PyTorch, OpenCV, Pandas, NumPy
Model: YOLOv10s
Dataset: VEDAI (COCO format)
Tools: Git, Jupyter Notebook

Installation

Clone the repository:
git clone https://github.com/your-username/Vehicle-Detection-YOLO.git
cd Vehicle-Detection-YOLO


Install dependencies:
pip install -r requirements.txt


Download the VEDAI dataset from Kaggle and place it in data/.

Place the trained model weights (best.pt) in models/. Pre-trained weights (yolov10s.pt) are available in models/.


Usage

Preprocess the dataset and train the model:
jupyter notebook src/preprocess.ipynb

Follow the notebook to convert COCO annotations and train YOLOv10s.

Run inference on a sample image:
python src/detect.py

Outputs are saved in outputs/.


Project Structure
Vehicle-Detection-YOLO/
├── src/                  # Source code
│   ├── preprocess.ipynb  # Data preprocessing and training
│   └── detect.py         # Inference script
├── data/                 # Dataset files
│   ├── vedai.yaml        # Dataset configuration
│   ├── train/            # Training images and labels
│   └── val/              # Validation images and labels
├── models/               # Model weights
│   ├── yolov10s.pt       # Pre-trained weights
│   └── best.pt           # Trained model
├── outputs/              # Results
│   ├── runs/             # Training logs
│   └── sample_output.jpg # Example detection
├── docs/                 # Documentation
│   ├── report.pdf        # Project report
│   └── screenshot.png    # Detection screenshot
├── requirements.txt      # Dependencies
├── README.md             # This file
└── LICENSE               # MIT License

Results

Trained YOLOv10s for 300 epochs, achieving:
Precision: 0.942
Recall: 0.989
mAP50: 0.995
mAP50:95: 0.979


Speed: 5.3ms inference per image on Tesla P100 GPU.

Screenshots
Contributions

Preprocessed VEDAI dataset from COCO to YOLO format.
Trained YOLOv10s model with high accuracy.
Developed inference script for real-time detection.
Documented project for reproducibility.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
Reach out via [your-email@example.com] or [LinkedIn profile URL] for questions or collaboration.
