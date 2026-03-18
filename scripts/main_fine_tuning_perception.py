import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time import sleep
from src.common.fine_tuning import FineTuner
from src.common.load_config_file import ConfigLoader
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
import sys
import os
import yaml

if __name__ == "__main__":

    config_file = "config.yaml"
    loader = ConfigLoader(config_file)
    config = loader.load()

# Load a model
    if len(sys.argv) > 2:
        use_case = sys.argv[1]
        cycle = sys.argv[2]

    else:
        cycle = None
    learning_rate = 0.01 / (10 if cycle is None else 10**int(cycle))
    DATA_SET_DIR = os.path.join("data", use_case, f"normal_{use_case}_test_manual_analysis", f"Rejected_{int(cycle)-1}")
    img_path = os.path.join(DATA_SET_DIR, "images")
    label_path = os.path.join(DATA_SET_DIR, "labels")
    if not os.path.exists(DATA_SET_DIR):
        os.makedirs(DATA_SET_DIR, exist_ok=True)
    print(f"**********************DATA_SET_DIR***********: {DATA_SET_DIR}")
    tuner = FineTuner(use_case, cycle)
    tuner.split_dataset(dataset_old=DATA_SET_DIR, img_path=img_path, label_path=label_path, mask_path=None, elite_path=None)
    print("Splitting done. Waiting for 0.5 seconds before creating YAML...")
    sleep(0.5)
    if use_case == "uc2":
        object_type = 'box'
    elif use_case == 'uc1':
        object_type = 'wood'
    yaml_data = {
    # Use `path` as dataset root and relative train/val so Ultraytics won't re-prepend the path
    'path': os.path.abspath(DATA_SET_DIR),
    'task': 'segment',  # segment for segmentation, obb for Oriented Bounding Box, aabb for Axis-Aligned Bounding Box
    'train': 'train/images',
    'val': 'val/images',
    'nc': 1,  # or the actual number of classes
    'names': [object_type]  # adapt to your case
    }

    yaml_file_path = os.path.join(DATA_SET_DIR, f"{use_case}_{object_type}.yaml")
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)
    if int(cycle) == 1:
        model_path = config['best_model'][f"{use_case}"]
    elif int(cycle) > 1:
        model_path = os.path.join("fine-tuning", use_case,  f"cycle{int(cycle)-1}_{use_case}", "weights", "best.pt")
    model = YOLO(model_path)  # YOLO pretrained model for segmentation
    # Train the model

    print(f"*********** Training with data config: {yaml_file_path}")
    model.train(data=yaml_file_path, epochs=50, imgsz=640, batch=16, lr0=learning_rate, name=f"cycle{cycle}_{use_case}", project=f"fine-tuning/{use_case}")  # train the model
    print("Model training completed.")
