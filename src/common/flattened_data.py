import pandas as pd
import json
import random
from .load_config_file import ConfigLoader
from .perception_difficulty import PerceptionDifficultyEvaluator
import os


class FeatureFlattener:
    """Flattens scene features from merged datasets into tabular CSV format."""

    def __init__(self, config_file: str, use_case: str):
        self.config_file = config_file
        self.use_case = use_case

    def flatten(self, cycle=None) -> tuple:
        """
        Flatten the scene features from the config file into a single dictionary.

        Returns:
            tuple: (df, dataset_output_path)
        """
        config = ConfigLoader(self.config_file).load()

        merged_data = config['merging'][f'{self.use_case}']['merged_data']
        print(f"Loaded scene configuration from: {self.config_file}")

        scene_features_file = os.path.join(f"{merged_data}_cycle{cycle}", f"cycle{cycle}_merged_stats.json")
        print(f"Loading scene features from: {scene_features_file}")

        with open(scene_features_file, 'r') as f:
            scene_features = json.load(f)

        all_images_data = []
        evaluator = PerceptionDifficultyEvaluator(self.config_file, self.use_case, cycle)

        for image_id, data in scene_features.items():
            row_data = {'image_id': image_id}
            poses = data['poses']
            if 'lighting' in data:
                row_data['lighting'] = data['lighting']

            for i, pose in enumerate(poses):
                obj_num = i + 1
                row_data[f'pos_x{obj_num}'] = pose['position'][0]
                row_data[f'pos_y{obj_num}'] = pose['position'][1]
                row_data[f'pos_z{obj_num}'] = pose['position'][2]
                row_data[f'rot_x{obj_num}'] = pose['orientation'][0]
                row_data[f'rot_y{obj_num}'] = pose['orientation'][1]
                row_data[f'rot_z{obj_num}'] = pose['orientation'][2]
                row_data[f'rot_w{obj_num}'] = pose['orientation'][3]

            image_path = data['image']
            label_path = data['label']
            row_data['difficulty'] = evaluator.evaluate_segmentation(image_path, label_path)
            all_images_data.append(row_data)

        out_put_temp = f"{self.use_case}_cycle{cycle}_flattened_scene_features.csv"
        dataset_output = os.path.join("data", self.use_case, out_put_temp)
        parent_dir = os.path.dirname(dataset_output)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        df = pd.DataFrame(all_images_data)
        with open(dataset_output, "w") as f:
            df.to_csv(f, index=False)

        print(f"Dataframe created successfully with shape: {df.shape}")
        return df, dataset_output
