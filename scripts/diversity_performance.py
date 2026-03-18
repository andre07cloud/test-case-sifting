import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu
import os
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
SAMPLE_SIZE = 10000 
RANDOM_SEED = 42

# =============================================================================
# UTILITAIRES
# =============================================================================

def cliffs_delta(lst1, lst2):
    """Calcule la taille d'effet Cliff's Delta."""
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    mat1 = np.array(lst1).reshape(-1, 1)
    mat2 = np.array(lst2).reshape(1, -1)
    diff = mat1 - mat2
    more = np.sum(diff > 0)
    less = np.sum(diff < 0)
    d = (more - less) / (m * n)
    return d

def get_feature_cols(df):
    """Récupère automatiquement les colonnes pos_ et rot_"""
    return [c for c in df.columns if c.startswith('pos_') or c.startswith('rot_')]

def compute_normalized_l2(df, features, sample_limit=SAMPLE_SIZE):
    """
    Calcule les distances L2 et les normalise par sqrt(N_dims)
    pour que le résultat soit toujours entre [0, 1].
    """
    data = df[features].values
    
    # Sampling
    if len(data) > sample_limit:
        indices = np.random.choice(len(data), sample_limit, replace=False)
        data = data[indices]
        
    # 1. Calcul Distance Euclidienne Brute
    dists = pdist(data, metric='euclidean')
    
    # 2. Normalisation par la distance max théorique
    # Si chaque feature est dans [0,1], la dist max est sqrt(N_features)
    n_features = data.shape[1]
    max_theoretical_dist = np.sqrt(n_features)
    
    if max_theoretical_dist > 0:
        dists_norm = dists / max_theoretical_dist
    else:
        dists_norm = dists
        
    return dists_norm

def interpret_effect_size(d):
    d = abs(d)
    if d < 0.147: return "Negligible"
    if d < 0.33: return "Small"
    if d < 0.474: return "Medium"
    return "Large"

def standardize_labels(df, col_name):
    """Force les labels à être 'Normal', 'Hard', 'Critical'."""
    def mapper(val):
        s = str(val).lower().strip()
        if s in ['1', '1.0', 'normal']: return 'Normal'
        if s in ['2', '2.0', 'hard']: return 'Hard'
        if s in ['3', '3.0', 'critical']: return 'Critical'
        return 'Unknown'
    df[col_name] = df[col_name].apply(mapper)
    return df

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_diversity(raw_path, selected_path, output_dir):
    print("--- Chargement des données ---")
    
    # 1. Chargement
    if not os.path.exists(raw_path):
        sys.exit(f"Erreur: Fichier RAW introuvable : {raw_path}")
    df_raw = pd.read_csv(raw_path)
    
    if not os.path.exists(selected_path):
        print(f"⚠️ Fichier Selected introuvable. Simulation...")
        df_selected = df_raw.sample(frac=0.1, random_state=42)
    else:
        df_selected = pd.read_csv(selected_path)
    
    cluster_col = 'difficulty_level'
    df_raw = standardize_labels(df_raw, cluster_col)
    df_selected = standardize_labels(df_selected, cluster_col)
    
    feature_cols = get_feature_cols(df_raw)
    print(f"Features détectées ({len(feature_cols)}) : {feature_cols[:3]}...")
    
    # --- SECURITE : GESTION DES NaN (DEMANDE UTILISATEUR) ---
    print("Application de fillna(0) sur les données manquantes...")
    df_raw[feature_cols] = df_raw[feature_cols].fillna(0)
    df_selected[feature_cols] = df_selected[feature_cols].fillna(0)

    # 2. Normalisation des Vecteurs [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_raw[feature_cols])
    
    df_raw_norm = df_raw.copy()
    df_selected_norm = df_selected.copy()
    
    df_raw_norm[feature_cols] = scaler.transform(df_raw[feature_cols])
    df_selected_norm[feature_cols] = scaler.transform(df_selected[feature_cols])
    
    stats_results = []
    plot_data = []

    target_order = ['Normal', 'Hard', 'Critical']
    
    for cluster_name in target_order:
        print(f"\n>>> Analyse du Cluster : {cluster_name} <<<")
        
        sub_raw = df_raw_norm[df_raw_norm[cluster_col] == cluster_name]
        sub_sel = df_selected_norm[df_selected_norm[cluster_col] == cluster_name]
        
        if len(sub_raw) < 2 or len(sub_sel) < 2:
            print(f"Pas assez de données pour {cluster_name}. Skip.")
            continue
            
        # 3. Calcul des Distances Normalisées [0, 1]
        dists_raw = compute_normalized_l2(sub_raw, feature_cols)
        dists_sel = compute_normalized_l2(sub_sel, feature_cols)
        
        mean_raw = np.mean(dists_raw)
        mean_sel = np.mean(dists_sel)
        
        stat, p_value = mannwhitneyu(dists_sel, dists_raw, alternative='greater')
        
        s_raw = np.random.choice(dists_raw, min(len(dists_raw), 10000), replace=False)
        s_sel = np.random.choice(dists_sel, min(len(dists_sel), 10000), replace=False)
        delta = cliffs_delta(s_sel, s_raw)
        effect = interpret_effect_size(delta)
        
        print(f"  Norm Mean: Unfiltered={mean_raw:.3f} -> Selected={mean_sel:.3f}")
        
        stats_results.append({
            'Cluster': cluster_name,
            'Mean_Unfiltered': mean_raw,
            'Mean_Selected': mean_sel,
            'Diversity_Gain_Pct': (mean_sel - mean_raw)/mean_raw*100 if mean_raw > 0 else 0,
            'P-Value': p_value,
            'Cliffs_Delta': delta,
            'Effect_Size': effect
        })
        
        # Sampling Plot
        plot_limit = 5000
        if len(dists_raw) > plot_limit: dists_raw = np.random.choice(dists_raw, plot_limit, replace=False)
        if len(dists_sel) > plot_limit: dists_sel = np.random.choice(dists_sel, plot_limit, replace=False)
        
        df_plot_raw = pd.DataFrame({'Distance': dists_raw, 'Group': 'Unfiltered', 'Cluster': cluster_name})
        df_plot_sel = pd.DataFrame({'Distance': dists_sel, 'Group': 'Selected', 'Cluster': cluster_name})
        plot_data.append(pd.concat([df_plot_raw, df_plot_sel]))

    if plot_data:
        full_plot_df = pd.concat(plot_data)
        
        # --- PLOT 1 : VIOLIN PLOT ---
        plt.figure(figsize=(12, 7))
        sns.violinplot(
            data=full_plot_df, 
            x='Cluster', 
            y='Distance', 
            hue='Group', 
            order=target_order,
            palette={'Unfiltered': '#d3d3d3', 'Selected': '#1f77b4'}, 
            split=False,
            inner="quart", 
            cut=0
        )
        plt.tick_params(axis='x', labelsize=14)
        plt.title('Intra-Cluster Diversity')
        plt.ylabel('Pairwise Distance')
        plt.xlabel('Difficulty Tier')
        plt.ylim(0, 1.05) 
        plt.legend(title='Dataset', loc='upper right')
        
        plt.tight_layout()
        save_path_violin = os.path.join(output_dir, "rq2_diversity_violin_L2.png")
        plt.savefig(save_path_violin, dpi=300)
        print(f"\n[1/2] Violin Plot sauvegardé : {save_path_violin}")

        # --- PLOT 2 : BOX PLOT ---
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=full_plot_df, 
            x='Cluster', 
            y='Distance', 
            hue='Group', 
            order=target_order,
            palette={'Unfiltered': '#d3d3d3', 'Selected': '#1f77b4'},
            showfliers=False
        )
        plt.title('Intra-Cluster Diversity')
        plt.ylabel('Pairwise Distance')
        plt.xlabel('Difficulty Tier')
        plt.ylim(0, 1.05)
        plt.legend(title='Dataset', loc='upper right')
        plt.tight_layout()
        save_path_box = os.path.join(output_dir, "rq2_diversity_boxplot_L2.png")
        plt.savefig(save_path_box, dpi=300)
        print(f"[2/2] Box Plot sauvegardé : {save_path_box}")

    # 6. Sauvegarde CSV
    df_stats = pd.DataFrame(stats_results)
    csv_path = os.path.join(output_dir, "rq2_diversity_stats_L2.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"Tableau Stats sauvegardé : {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        use_case = sys.argv[1]
        cycle = sys.argv[2]
    else:
        print("Usage: python analyze_diversity.py <uc1|uc2> <cycle_number>")
        use_case = "uc1"
        cycle = "0"
        print(f"Utilisation des valeurs par défaut: {use_case}, Cycle {cycle}")

    RAW_FILE = os.path.join("data", use_case, f"{use_case}_cycle{cycle}_features_with_clusters_kmeans.csv")
    SELECTED_FILE = os.path.join("data", use_case, f"cycle{cycle}_{use_case}_normal_features_diversifies_ANN_GLOBAL.csv")
    OUT_DIR = os.path.join("data", use_case, "rq2_diversity_analysis")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)
    
    analyze_diversity(RAW_FILE, SELECTED_FILE, OUT_DIR)