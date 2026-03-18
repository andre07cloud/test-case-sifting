import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.split_range_data import YAMEL_FILE
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
import sys


if __name__ == "__main__":

# Load a model
    model = YOLO("yolo11n-seg.pt")  # YOLO pretrained model for segmentation
    # Train the model
    data = f"data/{sys.argv[1]}_{YAMEL_FILE}"
    print(f"*********** Training with data config: {data}")
    model.train(data=data, epochs=50, imgsz=640, batch=16, name=f"train_results_{sys.argv[1]}", project=f"train_results/{sys.argv[1]}")  # train the model
    print("Model training completed.")
