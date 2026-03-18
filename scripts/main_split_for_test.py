import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.split_data_from_merge import MergedDatasetSplitter, OLD_DATASET_DIR
from src.common.load_config_file import ConfigLoader
import os
import sys
import yaml



if __name__ == "__main__":

    config_file = "config.yaml"
    loader = ConfigLoader(config_file)
    config = loader.load()

    use_case = sys.argv[1]
    cycle = sys.argv[2]
    merged_path = config['merging'][f'{use_case}']['merged_data']

    IMG_DIR = os.path.join(f"{merged_path}_cycle{cycle}", "images")
    LABEL_DIR = os.path.join(f"{merged_path}_cycle{cycle}", "labels")
    MASK_DIR = os.path.join(f"{merged_path}_cycle{cycle}", "masks")
    MASTER_JSON = os.path.join(f"{merged_path}_cycle{cycle}", f"cycle{cycle}_merged_stats.json")

    print("Splitting dataset merged for testing...")
    print("Use case:", sys.argv[1], type(sys.argv[1]))

    splitter = MergedDatasetSplitter(use_case)

    if os.path.exists(OLD_DATASET_DIR):
        splitter.split(img_path=IMG_DIR, label_path=LABEL_DIR, mask_path=MASK_DIR, cycle=cycle,
                       master_json=MASTER_JSON, dataset_old=OLD_DATASET_DIR)
    else:
        print(f"Directory {OLD_DATASET_DIR} does not exist. Please check the path.")
        splitter.split(img_path=IMG_DIR, label_path=LABEL_DIR, mask_path=MASK_DIR, cycle=cycle,
                       master_json=MASTER_JSON)
