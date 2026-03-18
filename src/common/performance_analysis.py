import pandas as pd
import yaml
import os
from ultralytics import YOLO
import json
import sys

from .load_config_file import ConfigLoader


class PerformanceAnalyzer:
    """Evaluates model performance per difficulty bucket across training cycles."""

    def __init__(self, config_file: str, use_case: str):
        self.config_file = config_file
        self.use_case = use_case

    def evaluate_per_bucket(self, mode: str, cycle: int = None):
        config = ConfigLoader(self.config_file).load()
        mode_cfg = config.get(mode)
        if mode_cfg is None:
            raise KeyError(f"Mode '{mode}' not found in configuration")
        main_use_case = mode_cfg.get('performences', {}).get(self.use_case)
        print(f"Main use: {main_use_case}")
        if main_use_case is None:
            raise KeyError(f"Use case '{self.use_case}' not found under mode '{mode}' in configuration")

        main_folder = f"{main_use_case['main_folder']}"
        main_dir_temp = f"{mode}_{self.use_case}_test_manual_analysis"
        print(f"Main dir temp: {main_dir_temp}")
        MAIN_DIR = os.path.join(main_folder, "data", f"{self.use_case}", f"{main_dir_temp}")
        print(f"Main directory for buckets: {MAIN_DIR}")

        if cycle and int(cycle) == 0:
            MODEL_PATH = config['best_model'][f'{self.use_case}']
            print(f"******** Model path for cycle 0: {MODEL_PATH}")
        elif cycle and int(cycle) >= 1:
            MODEL_PATH = os.path.join("fine-tuning", self.use_case, f"cycle{cycle}_{self.use_case}", "weights", "best.pt")
            print(f"Model path for cycle {cycle} >1: {MODEL_PATH}")

        DIFFICULTY_LEVEL = [1, 2, 3]
        CLASSES_DICT = {0: 'wood'}
        GLOBAL_PATH = f"{main_use_case['elite']}_{cycle}"
        NAIVE_PATH = f"{main_use_case['naive']}_cycle{cycle}"
        YAML_CONFIG_NAME = 'temp_bucket_data.yaml'
        ELITE_YAML_CONFIG_NAME = 'elite_data.yaml'
        NAIVE_YAML_CONFIG_NAME = 'naive_data.yaml'

        try:
            model = YOLO(MODEL_PATH)
        except Exception as e:
            print(f"ERREUR: Impossible de charger le modèle '{MODEL_PATH}'.")
            print("Veuillez vérifier le chemin de votre fichier best.pt.")
            sys.exit()

        metrics_dict = {
            'mAP50': {},
            'mAP50-95': {},
            'Precision': {},
            'Recall': {}
        }

        print(f"\n--- Démarrage de l'évaluation par Bucket ---")

        for level in DIFFICULTY_LEVEL:
            bucket_folder = os.path.join(MAIN_DIR, f'cycle{cycle}_{mode}_diversify_ANN_level_{level}')
            if not os.path.isdir(bucket_folder):
                print(f"  ATTENTION : Dossier {bucket_folder} introuvable. Skip...")
                continue
            print(f"\nÉvaluation du Niveau {level} : {bucket_folder}")

            yaml_config = {
                'path': bucket_folder,
                'train': '.',
                'val': 'images',
                'names': CLASSES_DICT
            }
            with open(YAML_CONFIG_NAME, 'w') as f:
                yaml.dump(yaml_config, f)
            print(f"  -> Fichier de configuration temporaire créé pour le niveau {level}.")

            try:
                metrics = model.val(data=YAML_CONFIG_NAME, verbose=True)
                metrics_dict['mAP50'][f'Difficulty_level {level}'] = metrics.seg.map50
                metrics_dict['mAP50-95'][f'Difficulty_level {level}'] = metrics.seg.map
                metrics_dict['Precision'][f'Difficulty_level {level}'] = metrics.seg.mp
                metrics_dict['Recall'][f'Difficulty_level {level}'] = metrics.seg.mr
            except Exception as e:
                print(f"  ERREUR lors de l'évaluation du niveau {level} : {e}")

        elite_yaml_config = {
            'path': GLOBAL_PATH,
            'train': '.',
            'val': 'images',
            'names': CLASSES_DICT
        }
        naive_yaml_config = {
            'path': NAIVE_PATH,
            'train': '.',
            'val': 'images',
            'names': CLASSES_DICT
        }
        with open(ELITE_YAML_CONFIG_NAME, 'w') as f:
            yaml.dump(elite_yaml_config, f)
        print(f"\nFichier de configuration elite créé : {ELITE_YAML_CONFIG_NAME}")

        with open(NAIVE_YAML_CONFIG_NAME, 'w') as f:
            yaml.dump(naive_yaml_config, f)
        print(f"\nFichier de configuration elite créé : {NAIVE_YAML_CONFIG_NAME}")

        for bucket in ['elite', 'naive']:
            if bucket == 'elite':
                current_yaml = ELITE_YAML_CONFIG_NAME
            else:
                current_yaml = NAIVE_YAML_CONFIG_NAME
            print(f"\n--- Évaluation du modèle sur le bucket '{bucket}' ---")
            try:
                metrics = model.val(data=current_yaml, verbose=True)
                print("#######################################################################")
                print("#######################################################################")

                metrics_dict['mAP50'][f'Overall_{bucket}'] = metrics.seg.map50
                metrics_dict['mAP50-95'][f'Overall_{bucket}'] = metrics.seg.map
                metrics_dict['Precision'][f'Overall_{bucket}'] = metrics.seg.mp
                metrics_dict['Recall'][f'Overall_{bucket}'] = metrics.seg.mr
                print(f"Dictionnary {metrics_dict}")
            except Exception as e:
                print(f"  ERREUR lors de l'évaluation du bucket {bucket} : {e}")

        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics = df_metrics.T
        df_metrics.index.name = 'Metrics'

        print(df_metrics)
        output_summary_dir = os.path.join(MAIN_DIR, f"cycle{cycle}")
        if not os.path.exists(output_summary_dir):
            print(f"Création du dossier : {output_summary_dir}")
            os.makedirs(output_summary_dir, exist_ok=True)

        summary_file_path = os.path.join(output_summary_dir, f'cycle{cycle}_{self.use_case}_{mode}_evaluation_rq1_summary.csv')
        df_metrics.to_csv(summary_file_path, index=True)

    @staticmethod
    def performance_summary(data_path: str) -> dict:
        """
        Analyse des performances globales à partir du fichier results.csv généré par YOLOv8.
        """
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()

        best_epoch_idx = df['metrics/mAP50-95(B)'].idxmax()
        best_metrics = df.loc[best_epoch_idx]
        json_performance = best_metrics.to_dict()

        print("\n")
        print(json_performance)
        print("--- Performance Globale Naïve (Validation Aléatoire) ---")
        print(f"Meilleure Époque : {int(best_metrics['epoch'])}")
        print(f"mAP50            : {best_metrics['metrics/mAP50(B)']:.5f}")
        print(f"mAP50-95         : {best_metrics['metrics/mAP50-95(B)']:.5f}")
        print(f"Précision        : {best_metrics['metrics/precision(B)']:.5f}")
        print(f"Rappel (Recall)  : {best_metrics['metrics/recall(B)']:.5f}")

        return json_performance

    @staticmethod
    def transpose(df):
        try:
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f"Erreur lors du chargement du fichier : {e}")
            sys.exit()

        metrics_cols = ['mAP50', 'mAP50-95', 'Precision', 'Recall']

        df_transposed = df.set_index('Difficulty_level')[metrics_cols].T
        df_transposed.columns = [f'Niveau {col}' for col in df_transposed.columns]

        naive_values = {}
        global_values = {}
        available_cols = df.columns.tolist()

        for metric in metrics_cols:
            naive_col_search = [c for c in available_cols if metric in c and 'Naïf' in c]
            if naive_col_search:
                naive_values[metric] = df[naive_col_search[0]].iloc[0]

            global_col_search = [c for c in available_cols if metric in c and ('overall' in c or 'elite' in c)]
            if global_col_search:
                global_values[metric] = df[global_col_search[0]].iloc[0]

        df_naif = pd.Series(naive_values, name='Naïf (Baseline)')
        df_global = pd.Series(global_values, name='Global Diversifié')

        df_final = pd.concat([df_transposed, df_global, df_naif], axis=1)
        df_final.index.name = 'Métrique'

        print("--- Résultat de la Transposition (Analyse RQ1) ---")
        print("Valeurs Naïves extraites :", naive_values)
        print("Valeurs Globales extraites :", global_values)
        print(df_final.round(4))

        output_file = 'uc2_evaluation_rq1_TRANSPOSED_FULL.csv'
        df_final.to_csv(output_file)
        print(f"\n Fichier transformé sauvegardé sous : {output_file}")
