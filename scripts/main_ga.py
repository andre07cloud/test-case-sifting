import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import sys
from src.problems.test_case_problem import *
from src.samplings.test_case_sampling import *
from src.common.test_crossover import TestCaseCrossover
from src.common.test_mutation import TestCaseMutation
from src.common.hierarchical_clustering import ClusteringEngine

from src.common.data_extraction import DataExtractor
from src.common.load_config_file import ConfigLoader
from src.common.calculate_apfd import APFDCalculator
from src.common.ga_algorithm import GeneticAlgorithmSolver
from src.common.compute_diversity import DiversityCalculator
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import matplotlib.patches as mpatches

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


#Random Search Method
def run_random_search(all_test_ids, data_map, n_evals):
    """
    Génère n_evals permutations.
    Retourne les métriques de la solution qui a la MEILLEURE DIFFICULTÉ (F1).
    Pourquoi ? Pour être comparable à la stratégie de sélection utilisée pour MOGA.
    """
    problem = TestCasePrioritizationProblem(all_test_ids, data_map)

    best_f1 = -np.inf
    best_f2 = -np.inf
    best_perm = None

    for _ in range(n_evals):
        perm = np.random.permutation(len(all_test_ids))

        out = {}
        problem._evaluate(np.array([perm]), out)

        # Pymoo retourne des valeurs négatives
        curr_f1 = -out["F"][0][0] # Difficulté (Positif)
        curr_f2 = -out["F"][0][1] # Diversité (Positif)

        # Stratégie de sélection : Maximiser F1 (Difficulté) prioritairement
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_f2 = curr_f2
            best_perm = [all_test_ids[i] for i in perm]

    return best_f1, best_f2, best_perm

def run_elite_ANN(all_test_ids, data_map, cluster_data, use_case):

    problem = TestCasePrioritizationProblem(all_test_ids, data_map)
    best_f1 = -np.inf
    best_f2 = -np.inf

    engine = ClusteringEngine(use_case)
    elite = engine.elite_search_ann(cluster_data)
    out = {}
    problem._evaluate(np.array([elite]), out)

    curr_f1 = -out["F"][0][0]
    curr_f2 = -out["F"][0][1]

    if curr_f1 > best_f1:
        best_f1 = curr_f1
        best_f2 = curr_f2
        best_perm = [all_test_ids[i] for i in elite]

    return best_f1, best_f2, best_perm

def get_metrics(indices, method_name, df, FAULT_PERCENTILE):
    """Calcule toutes les métriques pour un subset donné"""
    subset_df = df.iloc[indices]

    # 1. Reduction Rate
    orig_size = len(df)
    sub_size = len(subset_df)
    red_rate = 100 * (1 - sub_size / orig_size)

    # 2. Difficulty Score (Total)
    diff_score = subset_df['difficulty'].mean()

    # 3. APFD (Recall des fautes)
    apfd_calc = APFDCalculator(FAULT_PERCENTILE)
    apfd = apfd_calc.calculate(subset_df)

    return {
        "Method": method_name,
        "Reduction Rate (%)": red_rate,
        "Difficulty Score (Total)": diff_score,
        "APFD (Fault Recall)": apfd
    }


if __name__ == "__main__":

    config_file = "config.yaml"

    if len(sys.argv) > 1:
        use_case = sys.argv[1]
        mode = sys.argv[2]
    else:
        print("Usage: python main_performance.py <use_case> <test_bucket>")
        exit("Please provide a use case argument. uc1 or uc2 and test_bucket.")

    cluster_data = f"{use_case}_features_with_clusters_kmeans.csv"

    # --- 1. Chargement des données ---
    df = pd.read_csv(cluster_data)

    # --- 2. Préparation des Vecteurs (Features) ---
    feature_cols = [c for c in df.columns if c.startswith('pos_') or c.startswith('rot_')]

    # Remplacement des NaN par 0 (pour gérer les scènes avec moins d'objets)
    data = df[feature_cols].fillna(0).values.astype(np.float32)

    engine = ClusteringEngine(use_case)
    _, threshold, ketp_indices_ann, _, _ = engine.filter_dataset_ann(df, threshold=None, cluster="global", mode=mode)

    # Normalisation L2
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / np.where(norms == 0, 1, norms)

    # --- 3. Calcul de la Matrice de Distances ---
    dist_matrix = euclidean_distances(data, data)

    # --- 4. Initialisation des Matrices pour l'AG ---

    # A. Le Seuil (Threshold)
    THRESHOLD = threshold

    # B. CONFLICT_MATRIX
    CONFLICT_MATRIX = (dist_matrix < THRESHOLD).astype(int)
    print("Matrice de Conflits créée avec le seuil :", THRESHOLD)

    np.fill_diagonal(CONFLICT_MATRIX, 0)
    print("Exemple de la Matrice de Conflits (5x5) :\n", CONFLICT_MATRIX[:10, :10])
    # C. DIFFICULTIES
    DIFFICULTIES = df['difficulty'].values

    pop_size = 100
    N_EXPERIMENTS = 10   # Nombre de répétitions pour avoir des Boxplots significatifs
    POP_SIZE = 100
    N_GEN = 200           # Générations NSGA-II
    FAULT_PERCENTILE = 90
    results = []

    ga_solver = GeneticAlgorithmSolver(n_experiments=N_EXPERIMENTS, pop_size=POP_SIZE, n_gen=N_GEN)

    print(f"--- Benchmarking MOGA vs Random Search vs Elite ANN ({N_EXPERIMENTS} runs) ---")
    ann_metrics = get_metrics(ketp_indices_ann, "Elite ANN", df, FAULT_PERCENTILE)

    # On ajoute N fois pour avoir une ligne dans le boxplot (ou un point fixe)
    for _ in range(N_EXPERIMENTS):
        results.append(ann_metrics)

    for i in range(N_EXPERIMENTS):
        sys.stdout.write(f"\rRun {i+1}/{N_EXPERIMENTS}...")
        sys.stdout.flush()

        # 2. Random Search
        kept_indices_rs = ga_solver.random_search(n_iterations=N_GEN, data=data, dist_matrix=dist_matrix,
                                        difficulties=DIFFICULTIES, threshold=THRESHOLD)

        rs_metrics = get_metrics(kept_indices_rs, "Random Search", df, FAULT_PERCENTILE)
        results.append(rs_metrics)

        # --- A. MOGA (NSGA-II) ---
        _, kept_indice_ga, _, _, _, _ = ga_solver.solve_multi_objective(conflict_matrix=CONFLICT_MATRIX, difficulties=DIFFICULTIES)


        df_ga = df.iloc[kept_indice_ga]
        save_path = f"{use_case}_{mode}_moga_selected_tests_run{i+1}.csv"
        df_ga.to_csv(save_path, index=False)
        if len(kept_indice_ga) > 0:
            ga_metrics = get_metrics(kept_indice_ga, "MOGA (NSGA-II)", df, FAULT_PERCENTILE)
            results.append(ga_metrics)
        else:
            print("  -> MOGA n'a pas trouvé de solution valide.")

        _, kept_indice_soga, _, _, _, _ = ga_solver.solve_single_objective(conflict_matrix=CONFLICT_MATRIX, difficulties=DIFFICULTIES)
        df_soga = df.iloc[kept_indice_soga]
        save_path = f"{use_case}_{mode}_soga_selected_tests_run{i+1}.csv"
        df_soga.to_csv(save_path, index=False)
        if len(kept_indice_soga) > 0:
            soga_metrics = get_metrics(kept_indice_soga, "SOGA", df, FAULT_PERCENTILE)
            results.append(soga_metrics)
        else:
            print("  -> SOGA n'a pas trouvé de solution valide.")
    # --- 5. VISUALISATION ---
    df_res = pd.DataFrame(results)
    print("\n--- Résumé des Résultats Moyens ---")
    print(df_res.groupby("Method").mean())

    # Graphiques
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics_to_plot = ["Reduction Rate (%)", "Difficulty Score (Total)", "APFD (Fault Recall)"]
    colors = {'MOGA (NSGA-II)': '#1f77b4', 'Random Search': '#ff7f0e', 'Elite ANN': '#2ca02c', 'SOGA': '#d62728'}

    for ax, metric in zip(axes, metrics_to_plot):
        sns.boxplot(data=df_res, x='Method', y=metric, hue='Method', palette=colors, ax=ax, width=0.5)
        ax.set_title(metric, fontweight='bold')
        ax.legend([],[], frameon=False) # Cacher la légende interne

    plt.suptitle(f"Benchmark: MOGA vs Random vs Elite ANN vs SOGA ({N_EXPERIMENTS} runs)", fontsize=16)
    plt.tight_layout()
    plt.savefig("benchmark_comparison_final.png", dpi=300)
    plt.show()

    # Sauvegarde CSV
    df_res.to_csv(f"{use_case}_{mode}_benchmark_results_final.csv", index=True)
    print("Sauvegardé : benchmark_comparison_final.png et benchmark_results_final.csv")
