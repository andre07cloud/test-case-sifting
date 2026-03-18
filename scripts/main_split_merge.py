import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.split_range_data import DatasetSplitter, OLD_DATASET_DIR, YAMEL_FILE
from src.common.load_config_file import ConfigLoader
import os
import sys
import yaml



if __name__ == "__main__":

    use_case = sys.argv[1]
    cycle = sys.argv[2]
    if use_case == "uc2":
        object_type = 'box'
    elif use_case == 'uc1':
        object_type = 'wood'

    config_file = "config.yaml"
    loader = ConfigLoader(config_file)
    config = loader.load()

    train_path = config['merging'][f'{use_case}']['train_data']

    IMG_DIR = os.path.join(f"{train_path}_cycle{cycle}", "val", "images")
    LABEL_DIR = os.path.join(f"{train_path}_cycle{cycle}", "val", "labels")
    MASK_DIR = os.path.join(f"{train_path}_cycle{cycle}", "val", "masks")
    MASTER_JSON = os.path.join(f"{train_path}_cycle{cycle}", "val", "val_stats.json")

    splitter = DatasetSplitter(use_case)

    if os.path.exists(OLD_DATASET_DIR):
        splitter.split(img_path=IMG_DIR, label_path=LABEL_DIR, mask_path=MASK_DIR,
                       master_json=MASTER_JSON, dataset_old=OLD_DATASET_DIR)
    else:
        print(f"Directory {OLD_DATASET_DIR} does not exist. Please check the path.")
        splitter.split(img_path=IMG_DIR, label_path=LABEL_DIR, mask_path=MASK_DIR,
                       master_json=MASTER_JSON)



    # ++++++++++++++++++++
    # CREATE YAML
    # ++++++++++++++++++++
    root_path = f"{use_case}/{OLD_DATASET_DIR}"
    data = {
        'task': 'segment',  # segment for segmentation, obb for Oriented Bounding Box, aabb for Axis-Aligned Bounding Box
        'train': os.path.join(root_path, 'train/images'),
        'val': os.path.join(root_path, 'val/images'),
        'nc': 1,  # or the actual number of classes
        'names': [object_type]  # adapt to your case
    }

    with open(f"data/{use_case}_{YAMEL_FILE}", 'w') as f:
        yaml.dump(data, f)

    print(f"YAML file created at data/{use_case}_{YAMEL_FILE}")
