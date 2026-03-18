import os
import json
import shutil
import sys
import re

from .load_config_file import ConfigLoader


class DataMerger:
    """Merges multiple dataset directories into a single unified dataset."""

    def __init__(self, config_file: str, use_case: str):
        self.config_file = config_file
        self.use_case = use_case

    def merge(self, dir_count: int, cycle: int = None):
        print(f"--- Démarrage du Merging Dynamique pour {self.use_case} ---")

        # 1. Chargement de la configuration
        try:
            config_data = ConfigLoader(self.config_file).load()
            main_use_case = config_data['merging'][f'{self.use_case}']
            base_root_path = main_use_case['base_path']
            output_dir = f"{main_use_case['merged_data']}_cycle{cycle}"
        except KeyError as e:
            print(f"Erreur de configuration YAML: {e}")
            return

        # 2. Préparation des dossiers de sortie
        output_img_dir = os.path.join(output_dir, 'images')
        output_lbl_dir = os.path.join(output_dir, 'labels')
        ouput_mask_dir = os.path.join(output_dir, 'masks')

        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_lbl_dir, exist_ok=True)
        os.makedirs(ouput_mask_dir, exist_ok=True)
        print(f"Dossier de sortie prêt : {output_dir}")

        master_stats = {}
        global_id_counter = 0

        images_copied = 0
        labels_copied = 0
        masks_copied = 0
        for i in range(1, dir_count + 1):
            current_dir_name = f"dir_{i}"
            current_base_path = os.path.join(base_root_path, self.use_case, current_dir_name)
            json_filename = f"{current_dir_name}_stats.json"
            json_path = os.path.join(current_base_path, json_filename)

            print(f"\n--- Traitement de : {current_dir_name} ---")

            if not os.path.exists(json_path):
                print(f"  Fichier JSON introuvable : {json_path}. Passage au suivant.")
                continue

            try:
                with open(json_path, 'r') as f:
                    current_stats = json.load(f)
            except json.JSONDecodeError:
                print(f"  Erreur JSON dans {json_path}. Ignoré.")
                continue

            num_entries = len(current_stats)
            print(f"   -> {num_entries} entrées trouvées dans ce dossier.")

            sorted_keys = sorted(current_stats.keys(), key=lambda x: int(x) if x.isdigit() else x)

            files_copied_count = 0

            for original_key in sorted_keys:
                entry = current_stats[original_key]

                src_img_rel = entry.get('image')
                src_lbl_rel = entry.get('label')

                if not src_img_rel or not src_lbl_rel:
                    continue

                src_img_abs = os.path.join(current_base_path, src_img_rel)
                src_lbl_abs = os.path.join(current_base_path, src_lbl_rel)
                src_mask_abs = os.path.join(current_base_path, 'masks', f"{original_key}.png")

                if not os.path.exists(src_img_abs):
                    print(f"   Image manquante : {src_img_abs}")
                    continue

                new_id = str(global_id_counter)
                _, ext = os.path.splitext(src_img_rel)

                new_img_name = f"{new_id}{ext}"
                new_lbl_name = f"{new_id}.txt"
                new_mask_name = f"{new_id}.png"

                dst_img_abs = os.path.join(output_img_dir, new_img_name)
                dst_lbl_abs = os.path.join(output_lbl_dir, new_lbl_name)
                dst_mask_abs = os.path.join(ouput_mask_dir, new_mask_name)

                try:
                    shutil.copy2(src_img_abs, dst_img_abs)
                    images_copied += 1
                    if os.path.exists(src_lbl_abs):
                        shutil.copy2(src_lbl_abs, dst_lbl_abs)
                        labels_copied += 1
                    if os.path.exists(src_mask_abs):
                        shutil.copy2(src_mask_abs, dst_mask_abs)
                        masks_copied += 1

                    new_entry = entry.copy()
                    new_entry['image'] = f"images/{new_img_name}"
                    new_entry['label'] = f"labels/{new_lbl_name}"
                    master_stats[new_id] = new_entry
                    global_id_counter += 1
                    files_copied_count += 1

                except Exception as e:
                    print(f"   Erreur de copie pour {original_key}: {e}")

            print(f"   {files_copied_count} images fusionnées depuis {current_dir_name}.")

        merged_json_path = os.path.join(output_dir, f'cycle{cycle}_merged_stats.json')
        with open(merged_json_path, 'w') as f:
            json.dump(master_stats, f, indent=4)

        print(f"\n--- TERMINÉ ---")
        print(f"Total global : {global_id_counter} images.")
        print(f"Fichiers copiés : {images_copied} images, {labels_copied} labels, {masks_copied} masks.")
        print(f"JSON fusionné : {merged_json_path}")
