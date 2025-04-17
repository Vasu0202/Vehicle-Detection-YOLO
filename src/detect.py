
from ultralytics import YOLO
import cv2
import os

def run_inference(model_path, source, output_dir):
    # Load the trained model
    model = YOLO(model_path)
    
    # Run inference
    results = model(source)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and save results
    for i, result in enumerate(results):
        # Get the image with bounding boxes
        annotated_img = result.plot()
        
        # Save the output
        output_path = os.path.join(output_dir, f"output_{i}.jpg")
        cv2.imwrite(output_path, annotated_img)
        print(f"Saved output to {output_path}")

if __name__ == "__main__":
    model_path = "models/best.pt"
    source = "data/train/5690.jpg"  # Example image
    output_dir = "outputs"
    run_inference(model_path, source, output_dir)
```