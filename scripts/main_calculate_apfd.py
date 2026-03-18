import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
import sys
from src.common.calculate_apfd import APFDCalculator
from src.common.load_config_file import ConfigLoader

# --- PARAMÈTRES ---
# Fichier contenant les 1000 tests, leur difficulté et leur cluster
FILE_PATH = 'features_avec_clusters_hc.csv'


if __name__ == '__main__':

    config_file = "config.yaml"
    loader = ConfigLoader(config_file)
    config = loader.load()
    apfd_config = config['apfd']
    use_case = sys.argv[1]
    uc_apfd = apfd_config[f'{use_case}']
    print(f"######## {uc_apfd}   ###############")
    FAULT_PERCENTILE = apfd_config['fault_percentile']
    NAIVE_PATH = uc_apfd['naive']
    ELITE_PATH = uc_apfd['elite']

    calculator = APFDCalculator(FAULT_PERCENTILE)

    # --- 3. Génération des Ordres (P) et Calcul de l'APFD ---

    results = []

    # --- A. NAIF (Ensemble total sans filtre) ---
    # Ordre par défaut (image_id/index aléatoire)

    start_time = time.time()
    df = pd.read_csv(NAIVE_PATH)
    N_total = len(df)
    apfd_naive = calculator.calculate(df)
    results.append({'Method': 'Naïf (Random)', 'APFD (%)': apfd_naive, 'Nb Tests': N_total, 'Temps (s)': time.time() - start_time})

    # --- B. BUCKETS (Level 1, 2, 3) ---
    # Trié par difficulté décroissante pour la priorisation interne
    for level in [1, 2, 3]:
        level_path = uc_apfd[f'level_{level}']
        df_level = pd.read_csv(level_path)
        df_level = df_level.sort_values(by='difficulty', ascending=False)
        P_Level = df_level['image_id'].tolist()

        start_time = time.time()
        apfd_level = calculator.calculate(df_level)
        results.append({'Method': f'Bucket Level {level}', 'APFD (%)': apfd_level, 'Nb Tests': len(P_Level), 'Temps (s)': time.time() - start_time})


    # --- C. ELITE/GLOBAL DIVERSIFIÉ ---
    df = pd.read_csv(ELITE_PATH)
    df_elite = df.sort_values(by=['difficulty_level', 'difficulty'], ascending=[False, False])
    N_total = len(df_elite)
    P_Elite = df_elite['image_id'].tolist()
    start_time = time.time()
    apfd_elite = calculator.calculate(df_elite)
    results.append({'Method': 'Elite (Diversified, Filtred)', 'APFD (%)': apfd_elite, 'Nb Tests': N_total, 'Temps (s)': time.time() - start_time})


    # --- 4. Affichage et Sauvegarde ---
    df_results = pd.DataFrame(results)

    print("\n--- Résultat de l'Analyse APFD ---")
    print(df_results.round(3))

    # Sauvegarder pour comparaison future
    df_results.to_csv(f'{use_case}_apfd_analysis_summary.csv', index=False)
    print("\nFichier 'apfd_analysis_summary.csv' sauvegardé.")
