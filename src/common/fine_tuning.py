from .split_range_data import DatasetSplitter
import os
import random
import shutil
import sys
import json
import pandas as pd

TRAIN_RATIO = 0.8


class FineTuner:
    """Manages dataset splitting for fine-tuning cycles."""

    def __init__(self, use_case: str, cycle: int):
        self.use_case = use_case
        self.cycle = cycle

    def split_dataset(self, dataset_old, img_path, label_path, mask_path, elite_path):
        print("Creating dataset structure...")
        print(f"Using old dataset from: {dataset_old}")
        print(f"use case path: {self.use_case}")
        root_path = dataset_old
        train_img_dir = os.path.join(root_path, "train/images")
        train_label_dir = os.path.join(root_path, "train/labels")
        val_img_dir = os.path.join(root_path, "val/images")
        val_label_dir = os.path.join(root_path, "val/labels")

        for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
            os.makedirs(d, exist_ok=True)

        all_images = [f for f in os.listdir(img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(all_images)
        split_index = int(len(all_images) * TRAIN_RATIO)
        train_images = all_images[:split_index]
        val_images = all_images[split_index:]

        DatasetSplitter.move_files(files=train_images, img_dest=train_img_dir, label_dest=train_label_dir,
                                   img_src=img_path, label_src=label_path)

        master_json = os.path.join("data", self.use_case, f"normal_{self.use_case}_test_manual_analysis",
                                   f"Rejected_{int(self.cycle)-1}", f"{self.use_case}_rejected_data_features.json")
        print(f"master_stats: {master_json}")

        if master_json:
            try:
                with open(master_json, 'r') as f:
                    master_stats = json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement de : {e}")

            train_json_output = os.path.join(root_path, "train", "train_stats.json")
            DatasetSplitter.filter_and_save_json(train_images, master_stats, train_json_output)

        DatasetSplitter.move_files(files=val_images, img_dest=val_img_dir, label_dest=val_label_dir,
                                   img_src=img_path, label_src=label_path)

        if master_json:
            try:
                with open(master_json, 'r') as f:
                    master_stats = json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement de : {e}")

            val_json_output = os.path.join(root_path, "val", "val_stats.json")
            DatasetSplitter.filter_and_save_json(val_images, master_stats, val_json_output)

        print(f"Dataset created at {dataset_old} to Total {len(all_images)} with {len(train_images)} training and {len(val_images)} validation images.")
