import os
import random
import shutil
import yaml
import sys
import json


TRAIN_RATIO = 0.50
VAL_RATIO = 0.50
TEST_RATIO = 0.1
YAMEL_FILE = "wood.yaml"
OLD_DATASET_DIR = "dataset_original"


class MergedDatasetSplitter:
    """Splits a merged dataset into train/validation sets by cycle."""

    def __init__(self, use_case: str):
        self.use_case = use_case

    def split(self, img_path, label_path, mask_path, cycle, master_json, dataset_old=None):
        print("Creating dataset structure...")
        print(f"Using old dataset from: {dataset_old}")
        print(f"use case path: {self.use_case}")

        if dataset_old is None:
            dataset_old = OLD_DATASET_DIR

        root_path = f"data/{self.use_case}/{dataset_old}_cycle{cycle}"
        train_img_dir = os.path.join(root_path, "train/images")
        train_label_dir = os.path.join(root_path, "train/labels")
        val_img_dir = os.path.join(root_path, "val/images")
        val_label_dir = os.path.join(root_path, "val/labels")
        train_mask_dir = os.path.join(root_path, "train/masks")
        val_mask_dir = os.path.join(root_path, "val/masks")

        for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, train_mask_dir, val_mask_dir]:
            os.makedirs(d, exist_ok=True)

        all_images = [f for f in os.listdir(img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(all_images)
        all_masks = [f for f in os.listdir(mask_path) if f.lower().endswith('.png')]

        split_index = int(len(all_images) * TRAIN_RATIO)
        train_images = all_images[:split_index]
        train_masks = all_masks[:split_index]
        val_images = all_images[split_index:]
        val_masks = all_masks[split_index:]

        self.move_files(files=train_images, img_dest=train_img_dir, label_dest=train_label_dir,
                        img_src=img_path, label_src=label_path, mask_src=mask_path, mask_dest=train_mask_dir)

        if master_json:
            try:
                with open(master_json, 'r') as f:
                    master_stats = json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement de : {e}")

            train_json_output = os.path.join(root_path, "train", "train_stats.json")
            self.filter_and_save_json(train_images, master_stats, train_json_output)

        self.move_files(files=val_images, img_dest=val_img_dir, label_dest=val_label_dir,
                        img_src=img_path, label_src=label_path, mask_src=mask_path, mask_dest=val_mask_dir)

        if master_json:
            try:
                with open(master_json, 'r') as f:
                    master_stats = json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement de : {e}")

            val_json_output = os.path.join(root_path, "val", "val_stats.json")
            self.filter_and_save_json(val_images, master_stats, val_json_output)

        print(f"Dataset created at {dataset_old} to Total {len(all_images)} with {len(train_images)} training and {len(val_images)} validation images, and {len(train_masks)} training and {len(val_masks)} validation masks.")

    @staticmethod
    def filter_and_save_json(image_list, master_json_data, output_path):
        """
        Crée un sous-fichier JSON contenant uniquement les entrées correspondant aux images de la liste.
        """
        subset_data = {}

        for img_name in image_list:
            key = os.path.splitext(img_name)[0]

            if key in master_json_data:
                subset_data[key] = master_json_data[key]
            else:
                print(f"  Warning: Clé '{key}' non trouvée dans le JSON principal pour l'image {img_name}")
        print(f"  Filtrage terminé. {len(subset_data)} entrées trouvées pour {len(image_list)} images demandées.")
        with open(output_path, 'w') as f:
            json.dump(subset_data, f, indent=4)
        print(f"  JSON sauvegardé : {output_path} ({len(subset_data)} entrées)")

    @staticmethod
    def move_files(files, img_dest, label_dest, img_src, label_src, mask_src=None, mask_dest=None):
        for f in files:
            src_img = os.path.join(img_src, f)
            img_filename = os.path.basename(src_img)

            if label_src is not None:
                lbl_src = os.path.join(label_src, os.path.splitext(f)[0] + '.txt')
                lbl_dst = os.path.join(label_dest, os.path.splitext(f)[0] + '.txt')
                if os.path.exists(lbl_src):
                    shutil.copy(lbl_src, lbl_dst)

            img_dst = os.path.join(img_dest, f)
            shutil.copy(src_img, img_dst)

            if mask_src and mask_dest:
                mask_name = f"{os.path.splitext(f)[0]}.png"
                src_mask = os.path.join(mask_src, mask_name)
                dst_mask = os.path.join(mask_dest, mask_name)

                if os.path.exists(src_mask):
                    shutil.copy2(src_mask, dst_mask)
                else:
                    mask_name_alt = img_filename
                    src_mask_alt = os.path.join(mask_src, mask_name_alt)
                    if os.path.exists(src_mask_alt):
                        shutil.copy2(src_mask_alt, os.path.join(mask_dest, mask_name_alt))
