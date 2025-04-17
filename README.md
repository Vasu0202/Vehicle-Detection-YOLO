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


Results

Trained YOLOv10s for 300 epochs, achieving:
Precision: 0.942
Recall: 0.989
mAP50: 0.995
mAP50:95: 0.979


