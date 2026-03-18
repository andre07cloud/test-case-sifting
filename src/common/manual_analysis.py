import pandas as pd
import json
import os
import shutil
import sys
from .load_config_file import ConfigLoader


class ManualAnalyzer:
    """Organizes and manages elite/bucket datasets across training cycles."""

    def __init__(self, config_file: str, use_case: str):
        self.config_file = config_file
        self.use_case = use_case

    def analyze(self, mode: str, cycle: int = None):
        config = ConfigLoader(self.config_file).load()
        if not config:
            print("ERREUR : Impossible de charger le fichier de configuration.")
            return

        cycle_prefix = f"c{cycle}_"

        key_difficulty = config["difficulty_levels"]
        difficulty_levels = key_difficulty[f"{mode}"][f'{self.use_case}']

        csv_files = [
            difficulty_levels[0]['level_1'],
            difficulty_levels[1]['level_2'],
            difficulty_levels[2]['level_3']
        ]

        current_master_json_path = os.path.join(f"{config['master_json_path'][f'{self.use_case}']}_cycle{cycle}", f"cycle{cycle}_merged_stats.json")
        current_collected_folder = f"{config['collected_file_folder'][f'{self.use_case}']}_cycle{cycle}"

        out_put_temp = f"{mode}_{self.use_case}_{config.get('main_output_folder')}"
        base_output_folder = os.path.join("data", self.use_case, out_put_temp)

        current_elite_folder = os.path.join(base_output_folder, f"Elite_{cycle}")

        print(f"--- Démarrage Manual Analysis (Cycle {cycle}) ---")

        try:
            with open(current_master_json_path, 'r') as f:
                current_master_data = json.load(f)
            print(f"Master JSON (Cycle {cycle}) chargé.")
        except Exception as e:
            print(f"ERREUR Critique: Impossible de charger {current_master_json_path}: {e}")
            return

        elite_json_data = {}

        if cycle > 0:
            prev_cycle = cycle - 1
            prev_elite_folder = os.path.join(base_output_folder, f"Elite_{prev_cycle}")
            prev_elite_json_path = os.path.join(prev_elite_folder, f"{self.use_case}_elited_data_features.json")

            print(f"--- Héritage Elite : Copie Cycle {prev_cycle} -> Cycle {cycle} ---")

            if os.path.exists(prev_elite_folder):
                try:
                    shutil.copytree(prev_elite_folder, current_elite_folder, dirs_exist_ok=True)
                    if os.path.exists(prev_elite_json_path):
                        with open(prev_elite_json_path, 'r') as f:
                            elite_json_data = json.load(f)
                except Exception as e:
                    print(f"  ERREUR copie Elite: {e}")
            else:
                print("  Elite précédent introuvable.")
        else:
            print("--- Cycle 0 : Initialisation Elite ---")
            os.makedirs(current_elite_folder, exist_ok=True)

        os.makedirs(os.path.join(current_elite_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(current_elite_folder, "labels"), exist_ok=True)

        current_elite_json_path = os.path.join(current_elite_folder, f"{self.use_case}_elited_data_features.json")

        level_idx = 1
        new_items_count_elite = 0

        for csv_file in csv_files:
            actual_csv_path = os.path.join(csv_file, f"cycle{cycle}_{self.use_case}_{mode}_diverse_files_ann", f"{mode}_diversify_ANN_level_{level_idx}.csv")

            print(f"\n--- Traitement Niveau {level_idx} (CSV: {os.path.basename(actual_csv_path)}) ---")

            bucket_folder_name = f"cycle{cycle}_{mode}_diversify_ANN_level_{level_idx}"
            current_bucket_folder = os.path.join(base_output_folder, bucket_folder_name)
            current_bucket_json_path = os.path.join(current_bucket_folder, f"cycle{cycle}_{mode}_diversify_ANN_level_{level_idx}_features.json")

            bucket_json_data = {}

            if cycle > 0:
                print(f"Cycle {cycle} Fine-tuning - Héritage Bucket Niveau {level_idx}")
                prev_cycle = cycle - 1
                prev_bucket_folder_name = f"cycle{prev_cycle}_{mode}_diversify_ANN_level_{level_idx}"
                prev_bucket_folder = os.path.join(base_output_folder, prev_bucket_folder_name)
                prev_bucket_json_path = os.path.join(prev_bucket_folder, f"cycle{prev_cycle}_{mode}_diversify_ANN_level_{level_idx}_features.json")

                if os.path.exists(prev_bucket_folder):
                    try:
                        shutil.copytree(prev_bucket_folder, current_bucket_folder, dirs_exist_ok=True)
                        if os.path.exists(prev_bucket_json_path):
                            with open(prev_bucket_json_path, 'r') as f:
                                bucket_json_data = json.load(f)
                        print(f"  -> Héritage Bucket Niveau {level_idx} OK.")
                    except Exception as e:
                        print(f"  -> Erreur héritage Bucket: {e}")
            else:
                os.makedirs(current_bucket_folder, exist_ok=True)

            os.makedirs(os.path.join(current_bucket_folder, "images"), exist_ok=True)
            os.makedirs(os.path.join(current_bucket_folder, "labels"), exist_ok=True)

            if not os.path.exists(actual_csv_path):
                print("  CSV introuvable, passage au niveau suivant.")
                level_idx += 1
                continue

            try:
                df = pd.read_csv(actual_csv_path)
            except:
                level_idx += 1
                continue

            new_items_count_bucket = 0

            for index, row in df.iterrows():
                csv_unique_id = None
                if 'unique_id' in row:
                    csv_unique_id = str(row['unique_id'])

                if 'original_image_id' in row:
                    original_id = str(row['original_image_id'])
                elif 'image_id' in row:
                    original_id = str(row['image_id'])
                else:
                    continue

                if csv_unique_id:
                    if not csv_unique_id.startswith(cycle_prefix):
                        continue
                else:
                    if original_id not in current_master_data:
                        continue

                target_unique_id = f"{cycle_prefix}{original_id}"

                if target_unique_id in elite_json_data:
                    continue

                entry = current_master_data[original_id].copy()

                src_img_name = entry.get('image')
                src_lbl_name = entry.get('label')

                if src_img_name:
                    src_img_path = os.path.join(current_collected_folder, src_img_name)
                    ext = os.path.splitext(src_img_name)[1]
                    new_img_name = f"{target_unique_id}{ext}"

                    dest_elite_img = os.path.join(current_elite_folder, "images", new_img_name)
                    dest_bucket_img = os.path.join(current_bucket_folder, "images", new_img_name)

                    if os.path.exists(src_img_path):
                        shutil.copy(src_img_path, dest_elite_img)
                        shutil.copy(src_img_path, dest_bucket_img)

                        entry['image'] = new_img_name
                        entry['id'] = target_unique_id
                        new_items_count_elite += 1
                        new_items_count_bucket += 1
                    else:
                        continue

                if src_lbl_name:
                    src_lbl_path = os.path.join(current_collected_folder, src_lbl_name)
                    ext = os.path.splitext(src_lbl_name)[1]
                    new_lbl_name = f"{target_unique_id}{ext}"

                    dest_elite_lbl = os.path.join(current_elite_folder, "labels", new_lbl_name)
                    dest_bucket_lbl = os.path.join(current_bucket_folder, "labels", new_lbl_name)

                    if os.path.exists(src_lbl_path):
                        shutil.copy(src_lbl_path, dest_elite_lbl)
                        shutil.copy(src_lbl_path, dest_bucket_lbl)
                        entry['label'] = new_lbl_name

                elite_json_data[target_unique_id] = entry
                bucket_json_data[target_unique_id] = entry

            try:
                with open(current_bucket_json_path, 'w') as f:
                    json.dump(bucket_json_data, f, indent=4)
                print(f"  -> Bucket {level_idx} sauvegardé : {new_items_count_bucket} nouveaux items.")
            except Exception as e:
                print(f"  Erreur sauvegarde Bucket JSON: {e}")

            level_idx += 1

        try:
            with open(current_elite_json_path, 'w') as f:
                json.dump(elite_json_data, f, indent=4)
            print(f"\n Elite Global Cycle {cycle} sauvegardé.")
            print(f"   -> Total Entrées : {len(elite_json_data)}")
            print(f"   -> Nouveaux ajouts ce cycle : {new_items_count_elite}")
        except Exception as e:
            print(f"Erreur sauvegarde JSON Elite: {e}")

        rejected_json_data = {}
        rejected_json_path = os.path.join(base_output_folder, f"Rejected_{cycle}", f"{self.use_case}_rejected_data_features.json")
        rejected_df_path = os.path.join("data", f"{self.use_case}", f"cycle{cycle}_{self.use_case}_{mode}_features_REJECTED_ANN_GLOBAL.csv")
        rejected_images_output_dir = os.path.join(base_output_folder, f"Rejected_{cycle}", "images")
        rejected_labels_output_dir = os.path.join(base_output_folder, f"Rejected_{cycle}", "labels")
        os.makedirs(rejected_images_output_dir, exist_ok=True)
        os.makedirs(rejected_labels_output_dir, exist_ok=True)
        print(" --- Processing Rejected For Fine-Tuning ---")

        if os.path.exists(rejected_df_path):
            try:
                df_rej = pd.read_csv(rejected_df_path)
                rejected_json_data = {}
                rej_count = 0

                for index, row in df_rej.iterrows():
                    if 'original_image_id' in row:
                        original_id = str(row['original_image_id'])
                    elif 'image_id' in row:
                        original_id = str(row['image_id'])
                    else:
                        continue

                    unique_id = f"{cycle_prefix}{original_id}"

                    if original_id not in current_master_data:
                        continue
                    entry = current_master_data[original_id].copy()

                    src_img_name = entry.get('image')
                    if src_img_name:
                        src_path = os.path.join(current_collected_folder, src_img_name)
                        ext = os.path.splitext(src_img_name)[1]
                        new_name = f"{unique_id}{ext}"
                        dest_path = os.path.join(rejected_images_output_dir, new_name)

                        if os.path.exists(src_path):
                            shutil.copy(src_path, dest_path)
                            rej_count += 1
                            entry['image'] = new_name
                            entry['id'] = unique_id

                    src_label_name = entry.get('label')
                    if src_label_name:
                        src_path = os.path.join(current_collected_folder, src_label_name)
                        ext = os.path.splitext(src_label_name)[1]
                        new_name = f"{unique_id}{ext}"
                        dest_path = os.path.join(rejected_labels_output_dir, new_name)

                        if os.path.exists(src_path):
                            shutil.copy(src_path, dest_path)
                            entry['label'] = new_name
                    rejected_json_data[unique_id] = entry

                with open(rejected_json_path, 'w') as f:
                    json.dump(rejected_json_data, f, indent=4)
                print(f" Rejected JSON saved: {rejected_df_path} ({len(rejected_json_data)} entries)")
                print(f"{rej_count} images rejected copied and renamed")
            except Exception as e:
                print(f"Error in processing of Rejected: {e}")
        else:
            print(f" CSV File Rejected Not found: {rejected_df_path}")

        print("\n ETL successfuly finished ----------")
