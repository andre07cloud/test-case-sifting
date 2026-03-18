import pandas as pd
import os
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class GainCalculator:
    """Calculates performance gains and reduction factors between training cycles."""

    def __init__(self, use_case: str):
        self.use_case = use_case

    def calculate_gains(self, file_c0, file_c1, target_col='Overall_elite', cycle_start=None, cycle_end=None):
        """
        Calcule les gains de performance entre deux cycles d'évaluation.

        Args:
            file_c0 (str): Chemin vers le CSV du Cycle 0 (Baseline).
            file_c1 (str): Chemin vers le CSV du Cycle 1 (Fine-Tuned).
            target_col (str): La colonne à analyser (ex: 'Overall_elite' pour le Golden Set).
        """
        df_c0 = pd.read_csv(file_c0)
        df_c1 = pd.read_csv(file_c1)

        output_dir = os.path.join("data", self.use_case, "gain_analysis_results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        def get_metrics(df, col_name):
            df = df.set_index('Metrics')
            map50_95 = float(df.loc['mAP50-95', col_name])
            precision = float(df.loc['Precision', col_name])
            recall = float(df.loc['Recall', col_name])

            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            return map50_95, f1, precision, recall

        map_c0, f1_c0, p_c0, r_c0 = get_metrics(df_c0, target_col)
        map_c1, f1_c1, p_c1, r_c1 = get_metrics(df_c1, target_col)

        gain_map_abs = map_c1 - map_c0
        gain_f1_abs = f1_c1 - f1_c0

        gain_map_rel = (gain_map_abs / map_c0) * 100
        gain_f1_rel = (gain_f1_abs / f1_c0) * 100

        results = {
            f'Cycle_{cycle_start}': {'mAP50-95': map_c0, 'F1-Score': f1_c0, 'Precision': p_c0, 'Recall': r_c0},
            f'Cycle_{cycle_end}': {'mAP50-95': map_c1, 'F1-Score': f1_c1, 'Precision': p_c1, 'Recall': r_c1},
            'Gains Absolus': {'mAP50-95': gain_map_abs, 'F1-Score': gain_f1_abs},
            'Gains Relatifs (%)': {'mAP50-95': gain_map_rel, 'F1-Score': gain_f1_rel}
        }

        output_json = os.path.join(output_dir, f'{self.use_case}_gains_cycle_{cycle_start}_to_{cycle_end}.json')
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n Gains de performance sauvegardés dans : {output_json}\n")

        print(f"--- ANALYSE DE PERFORMANCE : {target_col} for {self.use_case} on {cycle_start} to {cycle_end} ---")
        print("-" * 50)
        print(f"{'Métrique':<15} | f'{'Cycle {cycle_start}':<10} | {'Cycle {cycle_end}':<10} | {'Gain Abs.':<10} | {'Gain Rel. (%)':<15}")
        print("-" * 50)
        print(f"{'mAP50-95':<15} | {map_c0:.4f}     | {map_c1:.4f}     | +{gain_map_abs:.4f}    | +{gain_map_rel:.2f}%")
        print(f"{'F1-Score':<15} | {f1_c0:.4f}     | {f1_c1:.4f}     | +{gain_f1_abs:.4f}    | +{gain_f1_rel:.2f}%")
        print("-" * 50)
        print(f"Détails Cycle {cycle_start}: Precision={p_c0:.4f}, Recall={r_c0:.4f}")
        print(f"Détails Cycle {cycle_end}: Precision={p_c1:.4f}, Recall={r_c1:.4f}")

    def calculate_reduction_factor(self, file_c0, file_c1, cycle_start=None, cycle_end=None):
        """
        Calcule le facteur de réduction des cas critiques entre deux cycles.
        """
        try:
            with open(file_c0, 'r') as f:
                stats_c0 = json.load(f)
            with open(file_c1, 'r') as f:
                stats_c1 = json.load(f)
        except FileNotFoundError:
            print("Erreur : Fichiers non trouvés. Vérifiez les chemins.")
            return

        output_dir = os.path.join("data", self.use_case, "gain_analysis_results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        critical_c0 = stats_c0.get('Count_Critical', 0)
        critical_c1 = stats_c1.get('Count_Critical', 0)

        print(f"--- ANALYSE DE LA RÉDUCTION DES CAS CRITIQUES ---")
        print(f"Cycle 0 (Baseline)   : {critical_c0} images critiques")
        print(f"Cycle 1 (Fine-Tuned) : {critical_c1} images critiques")

        factor = None
        percent_reduction = None

        if critical_c1 > 0:
            factor = critical_c0 / critical_c1
            percent_reduction = ((critical_c0 - critical_c1) / critical_c0) * 100

            print("-" * 40)
            print(f"SUMMARY RESULTS :")
            print(f"Facteur de réduction : {factor:.2f}x")
            print(f"Réduction en pourcentage : -{percent_reduction:.2f}%")
            print("-" * 40)
            print(f"...reducing the occurrence of critical edge cases by a factor of {factor:.1f}x.")

        elif critical_c0 > 0 and critical_c1 == 0:
            print("Facteur : Infini (Réduction totale des cas critiques !)")
        else:
            print("Pas de cas critiques détectés au départ ou données invalides.")

        results = {
            'Cycle_0_Critical': critical_c0,
            'Cycle_1_Critical': critical_c1,
            'Reduction_Factor': factor if critical_c1 > 0 else 'Infinity',
            'Percent_Reduction': percent_reduction if critical_c1 > 0 else 100.0
        }
        output_json = os.path.join(output_dir, f'{self.use_case}_reduction_factor_cycle_{cycle_start}_to_{cycle_end}.json')
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n Facteur de réduction sauvegardé dans : {output_json}\n")

    def plot_difficulty_and_apfd(self, cycle_start=0, cycle_end=3):
        dfs = {}
        base_path = os.path.join("data", self.use_case)
        for i in range(cycle_start, cycle_end + 1):
            filename = f"{self.use_case}_cycle{i}_features_with_clusters_kmeans.csv"
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                dfs[f"cycle{i}"] = pd.read_csv(file_path)
            else:
                print(f"  Fichier manquant: {file_path}")
                return

        combined_df = pd.DataFrame()
        for cycle, df in dfs.items():
            temp_df = df.copy()
            temp_df['Cycle'] = cycle
            combined_df = pd.concat([combined_df, temp_df])

        plt.figure(figsize=(10, 6))
        sns.violinplot(data=combined_df, x='Cycle', y='difficulty', palette="viridis", inner="quartile")
        plt.title('Distribution de la Difficulté par Cycle')
        plt.ylabel('Score de Difficulté')
        plt.xlabel('Cycle')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(base_path, f'{self.use_case}_violin_plot_difficulty.png'))

        for cycle in range(cycle_start, cycle_end + 1):
            df_cycle = dfs[f'cycle{cycle}'].copy()

            df_cycle['is_fault'] = df_cycle['difficulty_level'].isin([2, 3]).astype(int)
            df_cycle['is_critical'] = (df_cycle['difficulty_level'] == 3).astype(int)
            df_sorted = df_cycle.sort_values(by='difficulty', ascending=False).reset_index(drop=True)

            total_images = len(df_sorted)
            fraction_images = np.arange(1, total_images + 1) / total_images * 100

            cum_crit = np.cumsum(df_sorted['is_critical'])
            fraction_crit = cum_crit / df_sorted['is_critical'].sum()

            cum_faults = np.cumsum(df_sorted['is_fault'])
            fraction_faults = cum_faults / df_sorted['is_fault'].sum()

            n_simulations = 10
            sum_cum_random = np.zeros(total_images)
            total_faults_count = df_cycle['is_fault'].sum()

            print(f"Simulation de {n_simulations} tirages aléatoires en cours...")
            for i in range(n_simulations):
                df_shuffled = df_cycle.sample(frac=1, random_state=i).reset_index(drop=True)
                sum_cum_random += np.cumsum(df_shuffled['is_fault'])

            avg_cum_random = sum_cum_random / n_simulations
            fraction_random = avg_cum_random / total_faults_count

            plt.figure(figsize=(9, 6))

            plt.plot(fraction_images, fraction_crit * 100, label='Priority : Critical (Cluster Critical)',
                     linewidth=3, color='#d62728')
            plt.plot(fraction_images, fraction_faults * 100, label='Priority : All Faults (Clusters Critical + Hard)',
                     linewidth=2, color='#ff7f0e', linestyle='--')

            plt.plot(fraction_images, fraction_random * 100, linestyle=':', color='gray',
                     label=f'Random (Mean on {n_simulations} runs)', linewidth=2)

            idx_10pct = int(total_images * 0.10) - 1
            val_crit = fraction_crit[idx_10pct] * 100

            plt.scatter([10], [val_crit], color='black', zorder=5)
            plt.annotate(f"CRITICAL : {val_crit:.0f}% found\n(at 10% of images inspected)",
                         xy=(10, val_crit), xytext=(15, 90),
                         arrowprops=dict(facecolor='black', shrink=0.05))

            plt.title('APFD CURVE - Prioritizer vs Random')
            plt.xlabel('% of Images Inspected')
            plt.ylabel('% of Scenarios Found')
            plt.legend(loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.xlim(0, 100)
            plt.ylim(0, 105)

            apfd_path = os.path.join(base_path, f'{self.use_case}_apfd_curve_cycle{cycle}.png')
            plt.savefig(apfd_path)
            print(f"Plot saved at : {apfd_path}")
