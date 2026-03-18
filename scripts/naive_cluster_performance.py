import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import shutil
import yaml
import sys
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION (A MODIFIER SELON VOTRE ENVIRONNEMENT)
# =============================================================================
# 1. Le fichier CSV contenant les clusters (celui que vous avez joint)
CSV_PATH = "data/uc1/uc1_cycle0_features_with_clusters_kmeans.csv"

# 2. Dossier où se trouvent TOUTES vos images et labels originaux (Cycle 0)
SOURCE_IMAGES_DIR = "data/uc1/images/val"  # ou 'test' ou 'all'
SOURCE_LABELS_DIR = "data/uc1/labels/val"

# 3. Dossier de sortie (où les 3 sous-dossiers seront créés)
OUTPUT_BASE_DIR = "data"




# 5. Extension des images
IMG_EXT = ".png" 

# =============================================================================
# FONCTION 1 : ORGANISER LES DONNÉES PAR CLUSTER
# =============================================================================
def prepare_cluster_datasets(dataset, out_put_dir, use_case="uc1", cycle=0):
    config_file = yaml.safe_load(open("config.yaml"))
    
    merged_data = config_file['merging'][f"{use_case}"]['merged_data']
    SOURCE_IMAGES_DIR = os.path.join(f"{merged_data}_cycle{cycle}", "images")
    SOURCE_LABELS_DIR = os.path.join(f"{merged_data}_cycle{cycle}", "labels")
    print(f"--- Chargement du CSV : {dataset} ---")
    df = pd.read_csv(dataset)
    
    # Vérification des colonnes
    if 'Difficulty_level_KMeans' not in df.columns:
        # Fallback si on a seulement les entiers 1, 2, 3
        map_diff = {1: 'normal', 2: 'hard', 3: 'critical'}
        df['Difficulty_level_KMeans'] = df['difficulty_level'].map(map_diff)

    clusters = ['normal', 'hard', 'critical']
    
    # Nettoyage et Création des dossiers
    for cluster in clusters:
        cluster_dir = os.path.join(out_put_dir, cluster)
        if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir) # On nettoie pour être sûr
        
        os.makedirs(os.path.join(cluster_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(cluster_dir, 'labels'), exist_ok=True)

    print("--- Début de la copie des fichiers ---")
    counts = {c: 0 for c in clusters}
    missing = 0

    for _, row in df.iterrows():
        img_id = str(row['image_id'])
        cluster = row['Difficulty_level_KMeans'] # normal, hard, critical
        
        # Chemins Source
        src_img = os.path.join(SOURCE_IMAGES_DIR, img_id + IMG_EXT)
        src_lbl = os.path.join(SOURCE_LABELS_DIR, img_id + ".txt")
        
        # Chemins Destination
        dst_img = os.path.join(OUTPUT_BASE_DIR, cluster, 'images', img_id + IMG_EXT)
        dst_lbl = os.path.join(OUTPUT_BASE_DIR, cluster, 'labels', img_id + ".txt")
        
        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.copy(src_img, dst_img)
            shutil.copy(src_lbl, dst_lbl)
            counts[cluster] += 1
        else:
            # print(f"Warning: Missing file {img_id}")
            missing += 1

    print(f"Distribution créée : {counts}")
    print(f"Fichiers manquants : {missing}")
    return counts

# =============================================================================
# FONCTION 2 : ÉVALUER LE MODÈLE SUR CHAQUE CLUSTER
# =============================================================================
def evaluate_clusters(use_case="uc1", cycle=0):
    
    config_file = yaml.safe_load(open("config.yaml"))
    
    if cycle and int(cycle) == 0:
        MODEL_PATH = config_file['best_model'][f'{use_case}']
        print(f"******** Model path for cycle 0: {MODEL_PATH}")
    elif cycle and int(cycle) >= 1:
        MODEL_PATH = os.path.join("fine-tuning", use_case, f"cycle{cycle}_{use_case}", "weights", "best.pt")
        print(f"Model path for cycle {cycle} >1: {MODEL_PATH}")
    print(f"\n--- Chargement du modèle : {MODEL_PATH} ---")
    model = YOLO(MODEL_PATH)
    YAML_CONFIG_NAME = 'temp_bucket_data.yaml'
    CLASSES_DICT = {0: 'wood'} 

    results_summary = []
    
    clusters = ['normal', 'hard', 'critical']
    metrics_dict = {
        'mAP50': {},
        'mAP50-95': {},
        'Precision': {},
        'Recall': {}
    }
    
    for cluster in clusters:
        print(f"\n>>> Évaluation du cluster : {cluster.upper()} <<<")
        
        dataset_path = os.path.abspath(os.path.join(OUTPUT_BASE_DIR, cluster))
        
        yaml_config = {
            'path': dataset_path,
            'train': '.',
            'val': 'images',
            'names': CLASSES_DICT
        }
        with open(YAML_CONFIG_NAME, 'w') as f:
            yaml.dump(yaml_config, f)
        print(f"  -> Fichier de configuration temporaire créé pour le niveau {cluster}.")
      
        try:
            metrics = model.val(data=YAML_CONFIG_NAME, verbose=True)
           
            metrics_dict['mAP50'][f'{cluster}'] = metrics.seg.map50
            metrics_dict['mAP50-95'][f'{cluster}'] = metrics.seg.map
            metrics_dict['Precision'][f'{cluster}'] = metrics.seg.mp
            metrics_dict['Recall'][f'{cluster}'] = metrics.seg.mr
           # print(f"Dictionnary {metrics_dict}")
        except Exception as e:
            print(f"  ERREUR lors de l'évaluation du niveau {cluster} : {e}")
        
        
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.index.name = 'Tier'  
    df_metrics = df_metrics.reset_index()

    return df_metrics

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    
    if len(sys.argv) > 2:
        use_case = sys.argv[1]
        cycle = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        print("Usage: python main_performance.py <use_case> <test_bucket>")
        exit("Please provide a use case argument. uc1 or uc2 and test_bucket.")
    
    OUTPUT_BASE_DIR = os.path.join("data", use_case, f"cycle{cycle}_kmeans_clusters")
    DATA_CLUSTER_CSV = os.path.join("data", use_case, f"{use_case}_cycle{cycle}_features_with_clusters_kmeans.csv")
    
    # Etape 1 : Organiser les fichiers
    prepare_cluster_datasets(dataset=DATA_CLUSTER_CSV, out_put_dir=OUTPUT_BASE_DIR, use_case=use_case, cycle=cycle)
    
    # Etape 2 : Evaluer
    df_res = evaluate_clusters(use_case=use_case, cycle=cycle)
    
    # Etape 3 : Afficher le tableau final pour le papier
    print("\n\n================================================")
    print("TABLEAU POUR RQ1 (A copier-coller)")
    print("================================================")
    #df_res = pd.DataFrame(final_results)
    
    # Calcul de la dégradation par rapport au Normal
    normal_map = df_res.loc[df_res['Tier'] == 'normal', 'mAP50-95'].values[0]
    df_res['Degradation'] = df_res['mAP50-95'].apply(lambda x: f"{(x - normal_map) / normal_map * 100:.1f}%" if x != normal_map else "-")
    
    print(df_res.to_string(index=False))
    
    # Sauvegarde CSV
    df_res.to_csv(os.path.join(OUTPUT_BASE_DIR, f"{use_case}_cycle{cycle}_cluster_performance_results.csv"), index=False)
    print("\nRésultats sauvegardés dans 'cluster_performance_results.csv'")