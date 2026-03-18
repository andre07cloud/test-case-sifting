import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. HELPER FUNCTIONS (EXTRACTION & CALCUL)
# ==========================================

def get_metrics_from_csv(file_path, target_col='Overall_elite'):
    """Extrait mAP50-95 et calcule le F1-Score depuis le CSV."""
    try:
        df = pd.read_csv(file_path)
        # Nettoyage des noms de colonnes (parfois des espaces traînent)
        df['Metrics'] = df['Metrics'].astype(str).str.strip()
        df = df.set_index('Metrics')
        
        # Extraction sécurisée
        map50_95 = float(df.loc['mAP50-95', target_col])
        precision = float(df.loc['Precision', target_col])
        recall = float(df.loc['Recall', target_col])
        
        # Calcul F1
        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        return map50_95, f1
    except Exception as e:
        print(f"[ERR] CSV Read Error ({file_path}): {e}")
        return None, None

def get_critical_count_from_json(file_path):
    """Extrait le nombre de cas critiques depuis le JSON."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('Count_Critical', 0)
    except Exception as e:
        print(f"[ERR] JSON Read Error ({file_path}): {e}")
        return None

def calculate_gains(old_val, new_val):
    """Calcule le gain absolu et relatif."""
    if old_val is None or new_val is None: return 0, 0
    diff = new_val - old_val
    rel = (diff / old_val * 100) if old_val != 0 else 0
    return diff, rel

def calculate_reduction(old_count, new_count):
    """Calcule le facteur de réduction (ex: 8.7x)."""
    if old_count is None or new_count is None: return 0
    if new_count == 0: return float('inf') # Réduction totale
    return old_count / new_count

# ==========================================
# 2. PLOTTING FUNCTION (COMBO CHART)
# ==========================================

def plot_combo_chart(data, output_dir, use_case):
    """
    Génère le graphique dual-axis : Performance (Lignes) vs Critical (Barres).
    """
    cycles = [d['cycle'] for d in data]
    maps = [d['mAP'] for d in data]
    f1s = [d['F1'] for d in data]
    criticals = [d['critical_count'] for d in data]

    # Setup
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(11, 7))
    
    c_map = '#1f77b4'  # Bleu
    c_f1 = '#2ca02c'   # Vert
    c_crit = '#d62728' # Rouge

    # --- AXE DROIT (BARRES) ---
    ax2 = ax1.twinx()
    # Ajout d'un léger offset aux barres pour l'esthétique
    bars = ax2.bar(cycles, criticals, color=c_crit, alpha=0.25, width=0.5, label='Critical Failures')
    
    ax2.set_ylabel('Number of Critical Failures', color=c_crit, fontweight='bold', fontsize=12)
    ax2.tick_params(axis='y', labelcolor=c_crit)
    
    # Echelle dynamique pour laisser de la place aux flèches
    max_crit = max(criticals) if criticals else 10
    ax2.set_ylim(0, max_crit * 1.3)

    # Étiquettes sur les barres
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (max_crit*0.01),
                 f'{int(height)}', ha='center', va='bottom', color=c_crit, fontweight='bold')

    # --- AXE GAUCHE (LIGNES) ---
    line1, = ax1.plot(cycles, maps, color=c_map, marker='o', markersize=8, linewidth=3, label='mAP50-95')
    line2, = ax1.plot(cycles, f1s, color=c_f1, marker='s', markersize=8, linestyle='--', linewidth=2, label='F1-Score')

    ax1.set_ylabel('Model Performance (0-1)', color='black', fontweight='bold', fontsize=12)
    ax1.set_ylim(min(min(maps), min(f1s)) * 0.9, 1.02) # Zoom dynamique
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- ANNOTATIONS DE GAIN (GLOBAL: START -> END) ---
    if len(cycles) >= 2:
        # 1. Gain mAP
        start_map, end_map = maps[0], maps[-1]
        gain_map_pct = ((end_map - start_map) / start_map) * 100
        
        ax1.annotate(f'+{gain_map_pct:.1f}% Gain', 
                     xy=(cycles[-1], end_map), 
                     xytext=(cycles[-1], end_map - 0.05), # Position du texte
                     arrowprops=dict(arrowstyle='->', color=c_map, lw=2),
                     color=c_map, fontweight='bold', ha='center', fontsize=11, 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c_map, alpha=0.8))

        # 2. Réduction Critiques
        start_crit, end_crit = criticals[0], criticals[-1]
        if end_crit > 0:
            factor = start_crit / end_crit
            txt = f'-{factor:.1f}x Reduction'
        else:
            txt = "Total Elimination"
            
        # # Flèche courbe pour la réduction
        # ax2.annotate(txt, 
        #              xy=(cycles[-1], end_crit), 
        #              xytext=(len(cycles)/2, start_crit * 0.6), # Milieu du graph
        #              arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.2", color=c_crit, lw=2, ls='--'),
        #              color=c_crit, fontweight='bold', fontsize=11,
        #              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c_crit, alpha=0.8))

    # Légende et Titres
    lines = [line1, line2, bars]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=10)
    
    # plt.title(f'Continuous Improvement Analysis: {use_case.upper()}', pad=40, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarde
    plot_path = os.path.join(output_dir, f"{use_case}_performance_combo_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[PLOT] Graphique généré : {plot_path}")
    plt.close()

# ==========================================
# 3. MAIN LOGIC
# ==========================================

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    if len(sys.argv) < 2:
        use_case = "uc1" # Défaut
        print(f"Usage: python script.py <use_case> [cycle_start] [cycle_end]")
        print(f"Defaulting to use_case={use_case}")
    else:
        use_case = sys.argv[1]

    cycle_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    cycle_end = int(sys.argv[3]) if len(sys.argv) > 3 else 2 # Inclut jusqu'à cycle 2
    
    # Dossiers
    base_data_path = os.path.join("data", use_case, f"normal_{use_case}_test_manual_analysis")
    stats_data_path = os.path.join("data", use_case)
    output_dir = os.path.join("data", use_case, "gain_analysis_results")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Stockage des données
    full_report = {
        "use_case": use_case,
        "cycles_data": [],
        "gains_analysis": []
    }

    print(f"Running Analysis for {use_case} | Cycles {cycle_start} -> {cycle_end}")
    print("-" * 60)

    # --- BOUCLE DE COLLECTE ---
    raw_data_points = []

    for i in range(cycle_start, cycle_end + 1):
        c_name = f"Cycle {i}"
        
        # Construction des chemins
        # CSV: .../cycle0/cycle0_uc1_normal_evaluation_rq1_summary.csv
        file_csv = os.path.join(base_data_path, f"cycle{i}", f'cycle{i}_{use_case}_normal_evaluation_rq1_summary.csv')
        # JSON: .../uc1_cycle0_kmeans_difficulty_stats.json
        file_json = os.path.join(stats_data_path, f"{use_case}_cycle{i}_kmeans_difficulty_stats.json")

        # Extraction
        if os.path.exists(file_csv) and os.path.exists(file_json):
            map_val, f1_val = get_metrics_from_csv(file_csv)
            crit_val = get_critical_count_from_json(file_json)
            
            if map_val is not None and crit_val is not None:
                # Ajout aux données brutes pour le plot
                entry = {
                    "cycle": c_name,
                    "mAP": map_val,
                    "F1": f1_val,
                    "critical_count": crit_val
                }
                raw_data_points.append(entry)
                full_report["cycles_data"].append(entry)
                
                print(f"[{c_name}] OK -> mAP:{map_val:.3f} | F1:{f1_val:.3f} | Crit:{crit_val}")
            else:
                print(f"[{c_name}] FAIL -> Erreur lecture données.")
        else:
            print(f"[{c_name}] SKIP -> Fichiers manquants.")
            # print(f"   Missing: {file_csv}") 
            # print(f"   OR:      {file_json}")

    # --- BOUCLE DE CALCUL DE GAINS (Step-by-Step) ---
    print("-" * 60)
    for i in range(len(raw_data_points) - 1):
        curr = raw_data_points[i]
        next_ = raw_data_points[i+1]
        
        # Performance Gains
        g_map_abs, g_map_rel = calculate_gains(curr['mAP'], next_['mAP'])
        g_f1_abs, g_f1_rel = calculate_gains(curr['F1'], next_['F1'])
        
        # Reduction Factor
        red_factor = calculate_reduction(curr['critical_count'], next_['critical_count'])
        
        gain_entry = {
            "from_cycle": curr['cycle'],
            "to_cycle": next_['cycle'],
            "mAP_gain_abs": round(g_map_abs, 4),
            "mAP_gain_rel_percent": round(g_map_rel, 2),
            "F1_gain_abs": round(g_f1_abs, 4),
            "F1_gain_rel_percent": round(g_f1_rel, 2),
            "critical_reduction_factor": round(red_factor, 2)
        }
        full_report["gains_analysis"].append(gain_entry)
        
        print(f"Gain {curr['cycle']} -> {next_['cycle']}: mAP +{g_map_rel:.1f}% | Reduction: {red_factor:.1f}x")

    # --- SAUVEGARDE JSON ---
    json_path = os.path.join(output_dir, f"{use_case}_analysis_report.json")
    with open(json_path, 'w') as f:
        json.dump(full_report, f, indent=4)
    print(f"\n[SAVE] Rapport complet sauvegardé : {json_path}")

    # --- GENERATION DU PLOT ---
    if len(raw_data_points) > 0:
        plot_combo_chart(raw_data_points, output_dir, use_case)
    else:
        print("[WARN] Pas assez de données pour générer le graphique.")