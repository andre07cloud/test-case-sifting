import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import chebyshev as chebyshev_distances
from scipy.cluster import hierarchy as shc
import sys
import os
import json
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import joblib

try:
    import faiss
    HAS_FAISS = True
    print("Moteur ANN: FAISS (Mode Rapide)")
except ImportError:
    HAS_FAISS = False
    print("Moteur ANN: Numpy (Mode Compatibilité)")


class ClusteringEngine:
    """Provides hierarchical and k-means clustering, ANN-based diversity filtering, and elbow-based threshold analysis."""

    def __init__(self, use_case: str, plots_dir: str = "outputs/plots"):
        self.use_case = use_case
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def preprocess(self, df=None):
        """
        Applique la normalisation globale sur tout le dataset.
        1. MinMaxScaler : Met tout entre 0 et 1 (pour gérer les différences d'échelle Position vs Rotation).
        """
        if df is None:
            return None, None

        feature_cols = [c for c in df.columns if c.startswith('pos_') or c.startswith('rot_')]

        if not feature_cols:
            print("Erreur: Pas de features trouvées.")
            return df, []

        print(f"Pré-traitement global sur {len(df)} images et {len(feature_cols)} features...")

        data = df[feature_cols].fillna(0).values.astype('float32')

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        df[feature_cols] = data_scaled

        return df, feature_cols

    def kmeans_clustering(self, data, n_clusters=3, cycle=0):
        df = pd.read_csv(data)

        difficulties = df[['difficulty']].values

        kmeans_model_path = os.path.join("data", self.use_case, f"{self.use_case}_kmeans_reference_model.pkl")
        kmeans = None

        if int(cycle) == 0:
            print(f"--- Cycle 0: Training K-Means with K={n_clusters} reference model ---")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['Cluster_id_KMeans'] = kmeans.fit_predict(difficulties)
            joblib.dump(kmeans, kmeans_model_path)
            print(f"K-Means model saved at '{kmeans_model_path}'")
        else:
            print(f"--- Cycle {cycle}: Loading K-Means reference model ---")
            if os.path.exists(kmeans_model_path):
                kmeans = joblib.load(kmeans_model_path)
                df['Cluster_id_KMeans'] = kmeans.predict(difficulties)
                print(f"K-Means model loaded from '{kmeans_model_path}'")
            else:
                sys.exit(f"Erreur : Modèle K-Means de référence introuvable pour le Cycle {cycle}.")

        centroids = kmeans.cluster_centers_.flatten()
        print(f"\n--- Analyse pour K = {n_clusters} clusters ---")

        sorted_indices = np.argsort(centroids)
        mapping_labels = ['normal', 'hard', 'critical']

        mapping = {
            sorted_indices[i]: mapping_labels[i]
            for i in range(min(n_clusters, len(mapping_labels)))
        }

        df['Difficulty_level_KMeans'] = df['Cluster_id_KMeans'].map(mapping)
        difficulty_map = {'normal': 1, 'hard': 2, 'critical': 3}
        df['difficulty_level'] = df['Difficulty_level_KMeans'].map(difficulty_map).astype('Int64')

        print("Distribution des classes de difficulté :")
        counts_series = df["Difficulty_level_KMeans"].value_counts()
        print(counts_series)

        total_samples = len(df)
        stats_dict = {}

        for i, label in enumerate(mapping_labels):
            if i >= n_clusters:
                break

            count = int(counts_series.get(label, 0))
            if total_samples > 0:
                percentage = (count / total_samples) * 100
            else:
                percentage = 0.0

            label_cap = label.capitalize()
            stats_dict[f'Count_{label_cap}'] = count
            stats_dict[f'Percentage_{label_cap}'] = percentage

            current_centroid = centroids[sorted_indices[i]]
            stats_dict[f'Centroid_{label_cap}'] = float(current_centroid)

            subset = df[df['Difficulty_level_KMeans'] == label]

            if not subset.empty:
                min_val = subset['difficulty'].min()
                max_val = subset['difficulty'].max()
            else:
                min_val, max_val = 0.0, 0.0

            stats_dict[f'Range_{label_cap}'] = [float(min_val), float(max_val)]

        json_filename = f"{self.use_case}_cycle{cycle}_kmeans_difficulty_stats.json"
        json_output_path = os.path.join("data", self.use_case, json_filename)

        parent_dir = os.path.dirname(json_output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        try:
            with open(json_output_path, 'w') as json_file:
                json.dump(stats_dict, json_file, indent=4)
            print(f"\n K-Means stats saved to '{json_output_path}'")
        except Exception as e:
            print(f"Erreur sauvegarde JSON: {e}")

        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x='difficulty', hue='difficulty_level', palette='viridis', multiple='stack')
        plt.title(f'Difficulty Segmentation by K-Means (Cycle {cycle})')
        plt.xlabel('Difficulty Score')
        plt.ylabel('Number of Images')

        colors = ['blue', 'orange', 'red']
        for i in range(min(n_clusters, 3)):
            plt.axvline(x=centroids[sorted_indices[i]], color=colors[i], linestyle='--',
                        label=f'Centroid {mapping_labels[i].capitalize()}')

        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, f"{self.use_case}_cycle{cycle}_kmeans_difficulty_segmentation.png"))

        out_put_temp = f"{self.use_case}_cycle{cycle}_features_with_clusters_kmeans.csv"
        output_csv = os.path.join("data", self.use_case, out_put_temp)

        parent_dir = os.path.dirname(output_csv)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        df.to_csv(output_csv, index=False)
        print(f"\n Data with clusters saved on '{output_csv}'")

        return df, output_csv

    def hierarchical_clustering(self, data, method='ward', metric='euclidean'):
        df = pd.read_csv(data)

        difficulties = df['difficulty'].values
        if difficulties.ndim == 1:
            if difficulties.size < 2:
                raise ValueError("Need at least two observations for hierarchical clustering")
            if np.isnan(difficulties).any():
                raise ValueError("'difficulty' column contains NaN values; please clean the data before clustering")
            observations = difficulties.reshape(-1, 1)
        else:
            observations = difficulties

        plt.figure(figsize=(15, 8))
        plt.title("Dendrograms Hierarchical Clustering")
        plt.xlabel("size of clusters(number of images)")
        plt.ylabel("Distance(Variance of Wards' method)")
        Z = shc.linkage(observations, method=method, metric=metric)

        dend = shc.dendrogram(
            Z,
            truncate_mode='lastp',
            p=30,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            show_leaf_counts=True
        )

        example_cut_distance = 10
        plt.axhline(y=example_cut_distance, color='r', linestyle='--',
                    label=f'Ligne de coupe (exemple à y={example_cut_distance})')
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "dendrogramme_difficulte.png"))
        plt.show()

        print(f"\n Dendrogramme sauvegardé sous '{os.path.join(self.plots_dir, 'dendrogramme_difficulte.png')}'")

        K_choisi = 3
        clusters = shc.fcluster(Z, K_choisi, criterion='maxclust')
        df['Cluster_id_HC'] = clusters

        print(f"\n--- Analyse pour K = {K_choisi} clusters ---")
        cluster_analysis = df.groupby('Cluster_id_HC')['difficulty'].agg(
            Mean_difficulty='mean'
        ).sort_values(by='Mean_difficulty')
        print("\n***************Clusters sorting from(normal to critical) :")
        print(cluster_analysis)
        print("\nClusters sorting from(normal to critical) :")
        print(cluster_analysis)
        labels_difficulty = ['normal', 'hard', 'critical']

        mapping_dict = dict(zip(cluster_analysis.index, labels_difficulty))
        print(f"\nApplied Mapping : {mapping_dict}")

        df['Difficulty_label_level'] = df['Cluster_id_HC'].map(mapping_dict)
        difficulty_map = {'normal': 1, 'hard': 2, 'critical': 3}
        df['difficulty_level'] = df['Difficulty_label_level'].map(difficulty_map).astype('Int64')

        output_csv = f"{self.use_case}_features_with_clusters_hc.csv"
        print(df.head())
        df.to_csv(output_csv, index=False)
        print(f"\n Data with clusters saved on '{output_csv}'")

        return Z, output_csv

    def diversity_ann(self, df, mode, threshold=0.15, previous_df=None):
        cluster_col = 'difficulty_level'
        global_output_file = os.path.join("data", self.use_case, f"{self.use_case}_{mode}_features_diversifies_ANN_GLOBAL.csv")
        feature_cols = [c for c in df.columns if c.startswith('pos_') or c.startswith('rot_')]
        print(f"Features détectées : {len(feature_cols)} colonnes")
        df = df.sort_values(by='difficulty', ascending=False)
        print(f"Dataset trié par difficulté décroissante pour prioriser les tests difficiles.")
        print("**********************************")
        print(df['difficulty'].head())
        data = df[feature_cols].fillna(0).values.astype('float32')

        kept_indices = []
        rejected_indices = []

        print(f"Début du filtrage ANN (Seuil={threshold:.4f})...")

        if HAS_FAISS:
            d = data.shape[1]
            index = faiss.IndexFlat(d, faiss.METRIC_Linf)

            if previous_df is not None and not previous_df.empty:
                print("Chargement des vecteurs précédents pour comparaison...")
                try:
                    prev_data = previous_df[feature_cols].fillna(0).values.astype('float32')
                    index.add(prev_data)
                    print(f"  - {len(prev_data)} vecteurs précédents chargés dans l'index FAISS.")
                except Exception as e:
                    print(f"Erreur lors du chargement des données précédentes : {e}")
            else:
                print("Cold start without previous data.")

            count_rejected_by_previous = 0
            for i in range(len(data)):
                vec = data[i:i+1].astype('float32')
                if index.ntotal == 0:
                    index.add(vec)
                    kept_indices.append(i)
                else:
                    D, _ = index.search(vec, 1)
                    nearest_dist = D[0][0]

                    if nearest_dist > threshold:
                        index.add(vec)
                        kept_indices.append(i)
                    else:
                        rejected_indices.append(i)

        else:
            kept_vectors = []

            if previous_df is not None and not previous_df.empty:
                print("Chargement des vecteurs précédents pour comparaison...")
                try:
                    prev_data = previous_df[feature_cols].fillna(0).values.astype('float32')
                    for vec in prev_data:
                        kept_vectors.append(vec)
                    print(f"  - {len(prev_data)} vecteurs précédents chargés.")
                except Exception as e:
                    print(f"Erreur lors du chargement des données précédentes : {e}")

            for i in range(len(data)):
                vec = data[i]
                if not kept_vectors:
                    kept_vectors.append(vec)
                    kept_indices.append(df.index[i])
                else:
                    diffs = np.abs(np.array(kept_vectors) - vec)
                    dists = np.max(diffs, axis=1)
                    min_dist = np.min(dists)

                    if min_dist > threshold:
                        kept_vectors.append(vec)
                        kept_indices.append(df.index[i])
                    else:
                        rejected_indices.append(df.index[i])

        df_filtered = df.iloc[kept_indices]
        df_rejected = df.iloc[rejected_indices]
        reduction = 100 * (1 - len(kept_indices)/len(df))
        print(f"Filtrage terminé pour le cycle encours.")
        print(f"  - Cluster length Befor ANN: {len(df)} images")
        print(f"  - Previous Cluster length : {len(previous_df) if previous_df is not None else 0} images")
        print(f"  - Cluster length after ANN : {len(kept_indices)} images")
        print(f"  - Rejected Cluster length after ANN (R-)       : {len(rejected_indices)}")
        print(f"  - Reduction Rate : {reduction:.1f}%")

        return df_filtered, df_rejected, threshold, kept_indices

    def diversity_clustering(self, data, mode, metric='euclidean', threshold=0.45, cycle=0):
        cluster_col = 'difficulty_level'
        out_put_temp = f"cycle{cycle}_{self.use_case}_{mode}_diverse_files_ann"
        input_previous = f"cycle{int(cycle) - 1}_{self.use_case}_{mode}_diverse_files_ann" if int(cycle) > 0 else None
        output_folder = os.path.join("data", self.use_case, out_put_temp)
        global_temp = f"cycle{cycle}_{self.use_case}_{mode}_features_diversifies_ANN_GLOBAL.csv"
        global_temp_previous = f"cycle{int(cycle) -1}_{self.use_case}_{mode}_features_diversifies_ANN_GLOBAL.csv" if int(cycle) > 0 else "Not Exists"
        global_rejected_temp = f"cycle{cycle}_{self.use_case}_{mode}_features_REJECTED_ANN_GLOBAL.csv"
        df_previous = None

        global_rejected_file = os.path.join("data", self.use_case, global_rejected_temp)
        global_output_file = os.path.join("data", self.use_case, global_temp)
        global_output_file_previous = os.path.join("data", self.use_case, global_temp_previous)

        if global_temp_previous is not None and os.path.exists(global_output_file_previous):
            print("Loading existing Golden Data ...")
            df_previous = pd.read_csv(global_output_file_previous)
        else:
            print("Golden Data not exists ...")

        os.makedirs(output_folder, exist_ok=True)

        print(f"--- Starting diversification ANN (Faiss) ---")

        try:
            df = pd.read_csv(data)
        except FileNotFoundError:
            print(f"Erreur : Fichier '{data}' non trouvé.")
            sys.exit()
        except Exception as e:
            print(f"Erreur lors de la lecture du CSV : {e}")
            sys.exit()

        df, feature_cols = self.preprocess(df)
        df_previous, feature_cols_previous = self.preprocess(df_previous)

        print("Pré-traitement global appliqué aux features.")
        print("**********************************")

        feature_cols = [col for col in df.columns if (col.startswith('pos_') or col.startswith('rot_'))]

        if not feature_cols:
            print("Erreur : Aucune colonne 'pos_' or 'rot_' or 'lighting' trouvée.")
            sys.exit()

        print(f"Using {len(feature_cols)} feature columns (ex: {feature_cols[0]}, {feature_cols[1]}...).")

        all_rejected_dfs = []
        all_selected_dfs = []
        summary = {
            "output_folder": output_folder,
            "feature_columns_count": len(feature_cols),
            "clusters": []
        }

        unique_clusters = sorted(df[cluster_col].unique())

        for cluster_label in unique_clusters:
            print(f"\n--- Difficulty cluster processing : {cluster_label} ---")
            current_cycle_prefix = f"c{cycle}_"
            df_cluster = df[df[cluster_col] == cluster_label].copy()
            df_previous_cluster = df_previous[df_previous[cluster_col] == cluster_label].copy() if df_previous is not None else None
            if 'image_id' in df_cluster.columns:
                df_cluster['original_image_id'] = df_cluster['image_id'].astype(str)
                df_cluster['unique_id'] = current_cycle_prefix + df_cluster['image_id'].astype(str)
            else:
                print("Column 'image_id' Not found...")

            N = len(df_cluster)
            if metric == 'chebyshev':
                df_diverse, df_rejected, threshold, ketp_indices = self.diversity_ann(
                    df_cluster, threshold=threshold, mode=mode,
                    previous_df=df_previous_cluster)
            elif metric == 'euclidean':
                df_diverse, df_rejected, threshold, ketp_indices = self.filter_dataset_ann(
                    df_cluster, threshold=threshold, cluster=cluster_label, mode=mode)

            all_rejected_dfs.append(df_rejected)
            print(f"  Selecting {len(df_diverse)} diverse images (out of {N}).")
            df_diverse = df_diverse.sort_values(by='difficulty', ascending=False)
            print(f"********** Count selected **********: {len(df_diverse)} ")
            if df_previous is not None:
                print("******* Previous data cluster exists... **************** and ")
                df_golden_cluster = pd.concat([df_previous_cluster, df_diverse])
                print(f"*************Golden cluster: {len(df_golden_cluster)}")
                print(f"************* New Golden cluster size: {len(df_golden_cluster)}")
            else:
                print("previous data not exists...")
                df_golden_cluster = df_diverse

            df_golden_cluster.sort_values(by='difficulty', ascending=False)
            all_selected_dfs.append(df_golden_cluster)
            output_filename = os.path.join(output_folder, f"{mode}_diversify_ANN_level_{cluster_label}.csv")
            df_golden_cluster.to_csv(output_filename, index=False)
            print(f"  -> File saved : {output_filename}")

            try:
                cluster_label_int = int(cluster_label)
            except Exception:
                cluster_label_int = cluster_label

            summary["clusters"].append({
                "distance_threshold": threshold,
                "cluster_label": cluster_label_int,
                "cluster_length": int(len(df_cluster)),
                "Cluster_selected_Count": int(len(df_diverse)),
                "cluster_rejected_Count": int(len(df_rejected)),
                "Golden_selected_count": int(len(df_golden_cluster)),
                "Cluster_Reduction_Percentage": f"{float(100 * (1 - len(df_diverse)/len(df_cluster))):.2f}%",
                "Golden_Reduction_Percentage": f"{float(100 * (1 - len(df_golden_cluster)/len(df_cluster))):.2f}%",
                "output_file": output_filename
            })

        if all_selected_dfs:
            difficulty_order = {'normal': 1, 'hard': 2, 'critical': 3}
            df_global = pd.concat(all_selected_dfs)
            print(f"\nTotal images selected across all clusters: {len(df_global)}")
            df_global['difficulty_order'] = df_global['difficulty_level'].map(difficulty_order)
            df_global = df_global.sort_values(by=['difficulty_order', 'difficulty'], ascending=[False, False])
            df_global = df_global.drop('difficulty_order', axis=1)
            parent_dir = os.path.dirname(global_output_file)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            df_global.to_csv(global_output_file, index=False)
            print(f"\n GLOBAL file saved ({len(df_global)} images) : '{global_output_file}'")

        if all_rejected_dfs:
            difficulty_order = {'normal': 1, 'hard': 2, 'critical': 3}
            df_global_rejected = pd.concat(all_rejected_dfs)

            if 'difficulty_level' in df_global_rejected.columns:
                df_global_rejected['difficulty_order'] = df_global_rejected['difficulty_level'].map(difficulty_order)
                df_global_rejected = df_global_rejected.sort_values(by=['difficulty_order', 'difficulty'], ascending=[False, False])
                df_global_rejected = df_global_rejected.drop('difficulty_order', axis=1)

            if parent_dir_rejected := os.path.dirname(global_rejected_file):
                os.makedirs(parent_dir_rejected, exist_ok=True)
            df_global_rejected.to_csv(global_rejected_file, index=False)
            print(f"\n[R-] GLOBAL REJECTED file saved ({len(df_global_rejected)} images) : '{global_rejected_file}'")

            summary["global"] = {
                "metric": metric,
                "global_output_file": global_output_file,
                "global_selected_count": int(len(df_global)),
                "global_rejected_count": int(len(df_global_rejected))
            }

            summary_path = os.path.join(output_folder, f"cycle{cycle}_{self.use_case}_{mode}_diversify_summary.json")
            try:
                with open(summary_path, 'w', encoding='utf-8') as jf:
                    json.dump(summary, jf, ensure_ascii=False, indent=2)
                print(f"\n JSON summary saved: '{summary_path}'")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde du JSON summary : {e}")
        else:
            print("\nNo data was processed.")
        return global_output_file, global_rejected_file

    def filter_dataset_ann(self, df, threshold=None, cluster=None, mode=None):
        """
        Filtre le dataset en ne gardant que les images distantes d'au moins 'threshold'.
        """
        feature_cols = [c for c in df.columns if c.startswith('pos_') or c.startswith('rot_')]
        print(f"Features détectées : {len(feature_cols)} colonnes")
        df = df.sort_values(by='difficulty', ascending=False)
        print(f"Dataset trié par difficulté décroissante pour prioriser les tests difficiles.")
        print("**********************************")
        data = df[feature_cols].fillna(0).values.astype('float32')

        kept_indices = []

        print(f"Début du filtrage ANN (Seuil={threshold:.2f})...")

        if HAS_FAISS:
            d = data.shape[1]
            index = faiss.IndexFlatL2(d)

            for i in range(len(data)):
                vec = data[i:i+1]
                if index.ntotal == 0:
                    index.add(vec)
                    kept_indices.append(i)
                else:
                    D, _ = index.search(vec, 1)
                    nearest_dist = np.sqrt(D[0][0])
                    if nearest_dist > threshold:
                        index.add(vec)
                        kept_indices.append(i)
                        print(f"Image {i} ajoutée (Distance au plus proche : {nearest_dist:.4f})")
        else:
            kept_vectors = []
            for i in range(len(data)):
                vec = data[i]
                if not kept_vectors:
                    kept_vectors.append(vec)
                    kept_indices.append(i)
                else:
                    dots = np.dot(kept_vectors, vec)
                    dots = np.clip(dots, -1.0, 1.0)
                    dists = np.sqrt(2 - 2 * dots)
                    min_dist = np.min(dists)

                    if min_dist > threshold:
                        kept_vectors.append(vec)
                        kept_indices.append(i)

        all_indices = set(df.index)
        kept_set = set(kept_indices)
        rejected_indices = list(all_indices - kept_set)
        df_filtered = df.iloc[kept_indices]
        df_rejected = df.loc[rejected_indices]
        reduction = 100 * (1 - len(kept_indices)/len(df))
        print(f"Filtrage terminé.")
        print(f"  - Avant : {len(df)} images")
        print(f"  - Après : {len(kept_indices)} images")
        print(f"  - Réduction : {reduction:.1f}%")

        return df_filtered, df_rejected, threshold, kept_indices

    def diverse_before_clustering(self, data, mode, metric='euclidean', threshold=0.45):
        df = pd.read_csv(data)
        cluster_col = 'difficulty_level'
        out_put_temp = f"{self.use_case}_{mode}_diverse_files_ann"
        output_folder = os.path.join("data", self.use_case, out_put_temp)
        global_temp = f"{self.use_case}_{mode}_features_diversifies_ANN_GLOBAL.csv"
        global_output_file = os.path.join("data", self.use_case, global_temp)

        os.makedirs(output_folder, exist_ok=True)

        df, feature_cols = self.preprocess(df)

        summary = {
            "metric": metric,
            "output_folder": output_folder,
            "feature_columns_count": len(feature_cols),
            "clusters": []
        }
        all_selected_dfs = []
        print("Pré-traitement global appliqué aux features.")
        print("**********************************")
        print(df[feature_cols].head())
        df_cluster = df.copy()

        N = len(df_cluster)
        if metric == 'chebyshev':
            df_diverse, df_rejected, threshold, ketp_indices = self.diversity_ann(df_cluster, threshold=threshold, mode=mode)
        elif metric == 'euclidean':
            df_diverse, df_rejected, threshold, ketp_indices = self.filter_dataset_ann(df_cluster, threshold=threshold, mode=mode)

        df_diverse = df_diverse.sort_values(by='difficulty', ascending=False)

        summary["clusters"].append({
            "distance_threshold": threshold,
            "cluster_label": "global",
            "cluster_length": int(len(df_cluster)),
            "selected_count": int(len(df_diverse)),
            "Percentage Reduction": f"{float(100 * (1 - len(df_diverse)/len(df_cluster))):.2f}%",
        })

        difficulty_order = {'normal': 1, 'hard': 2, 'critical': 3}
        df_global = df_diverse.copy()
        df_global['difficulty_order'] = df_global['difficulty_level'].map(difficulty_order)
        df_global = df_global.sort_values(by=['difficulty_order', 'difficulty'], ascending=[False, False])
        df_global = df_global.drop('difficulty_order', axis=1)
        parent_dir = os.path.dirname(global_output_file)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        df_global.to_csv(global_output_file, index=False)
        print(f"\n GLOBAL file saved ({len(df_global)} images) : '{global_output_file}'")

        unique_clusters = sorted(df_diverse[cluster_col].unique())
        for cluster_label in unique_clusters:
            print(f"\n--- Difficulty cluster processing : Cluster {cluster_label} ---")
            df_cluster = df_diverse[df_diverse[cluster_col] == cluster_label].copy()
            N = len(df_diverse)

            print(f"  Selecting {len(df_cluster)} diverse images (out of {N}).")
            df_cluster = df_cluster.sort_values(by='difficulty', ascending=False)
            output_filename = os.path.join(output_folder, f"{mode}_diversify_ANN_level_{cluster_label}.csv")
            df_cluster.to_csv(output_filename, index=False)
            print(f"  -> File saved : {output_filename}")

            try:
                cluster_label_int = int(cluster_label)
            except Exception:
                cluster_label_int = cluster_label

            summary["clusters"].append({
                "cluster_label": cluster_label_int,
                "cluster_length": int(len(df_cluster)),
                "selected_count": int(len(df_diverse)),
                "output_file": output_filename
            })
        print(summary)
        summary_path = os.path.join(output_folder, f"{metric}_{self.use_case}_{mode}_diversify_summary.json")
        try:
            with open(summary_path, 'w', encoding='utf-8') as jf:
                json.dump(summary, jf, ensure_ascii=False, indent=2)
            print(f"\n JSON summary saved: '{summary_path}'")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du JSON summary : {e}")

        return global_output_file

    def get_elbow_threshold(self, data, cluster, mode) -> float:
        """
        Calcule le seuil optimal via la méthode du Coude (Elbow Method)
        et génère un graphique justificatif pour publication scientifique.
        """
        print("Calcul du seuil optimal (Elbow Method)...")
        plot_filename = os.path.join(self.plots_dir, f"{self.use_case}_{mode}_{cluster}_elbow_method_proof.png")
        nbrs = NearestNeighbors(n_neighbors=2).fit(data)
        distances, _ = nbrs.kneighbors(data)
        curve = np.sort(distances[:, 1])

        n_points = len(curve)
        all_coords = np.vstack((range(n_points), curve)).T
        first_point = all_coords[0]
        last_point = all_coords[-1]
        line_vec = last_point - first_point
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

        vec_from_first = all_coords - first_point
        scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))

        idx_elbow = np.argmax(dist_to_line)
        threshold = curve[idx_elbow]
        percentile = 100 * idx_elbow / n_points

        print(f"  -> Coude détecté au percentile {percentile:.1f}%")
        print(f"  -> Seuil (epsilon) calculé : {threshold:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(curve, color='#D59C28', linewidth=2, label='Distances k-NN triées')
        plt.axvline(x=idx_elbow, color='black', linestyle='--', alpha=0.7)
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
                    label=f'Seuil optimal ($\\epsilon={threshold:.3f}$)')
        plt.scatter(idx_elbow, threshold, color='red', s=100, zorder=5,
                    label='Point d\'inflexion (Elbow)')
        plt.title(f"Détermination du seuil de diversité (Elbow Method)\nPercentile: {percentile:.1f}%")
        plt.xlabel("Index des images (triées par similarité)")
        plt.ylabel("Distance au voisin le plus proche")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  -> Graphique justificatif sauvegardé sous : {plot_filename}")
        plt.close()

        return threshold

    def elite_search_ann(self, data_path):
        """
        Sélectionne un sous-ensemble diversifié en utilisant l'ANN (Faiss).
        Gère les NaNs, privilégie les tests difficiles et retourne les métriques F1/F2.
        """
        DISTANCE_THRESHOLD = 5.0
        K_NEIGHBORS = 50

        output_folder = f"{self.use_case}_diverse_files_ann"
        global_output_file = f"{self.use_case}_features_diversifies_ANN_GLOBAL.csv"
        os.makedirs(output_folder, exist_ok=True)

        print(f"--- Starting Improved Diversification ANN (Faiss) ---")

        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            sys.exit(f"Erreur lecture CSV: {e}")

        if 'lighting' in df.columns:
            df['lighting'] = np.log1p(df['lighting'])

        feature_cols = [col for col in df.columns if (col.startswith('pos_') or col.startswith('rot_') or col == 'lighting')]

        if not feature_cols:
            sys.exit("Erreur : Aucune colonne de features trouvée.")

        print(f"Features detected: {len(feature_cols)} columns.")

        df[feature_cols] = df[feature_cols].fillna(0.0)

        scaler = StandardScaler()
        features_all_scaled = scaler.fit_transform(df[feature_cols].values).astype('float32')
        df['features_vec'] = list(features_all_scaled)

        cluster_col = 'difficulty_level'
        if cluster_col not in df.columns:
            df[cluster_col] = 'default'

        unique_clusters = sorted(df[cluster_col].unique())
        all_selected_indices = []

        for cluster_label in unique_clusters:
            print(f"\nProcessing Cluster: {cluster_label}")

            df_cluster = df[df[cluster_col] == cluster_label].copy()
            if df_cluster.empty:
                continue

            df_cluster = df_cluster.sort_values(by='difficulty', ascending=False)
            features_cluster = np.stack(df_cluster['features_vec'].values)
            indices_cluster = df_cluster.index.tolist()

            d = features_cluster.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(features_cluster)

            N = len(features_cluster)
            used_mask = np.zeros(N, dtype=bool)
            k_search = min(N, K_NEIGHBORS)

            selected_local_indices = []

            for i in range(N):
                if not used_mask[i]:
                    selected_local_indices.append(i)
                    query_vec = features_cluster[i].reshape(1, -1)
                    D, I = index.search(query_vec, k=k_search)
                    neighbors_indices = I[0]
                    neighbors_dists = D[0]
                    close_neighbors = neighbors_indices[neighbors_dists < DISTANCE_THRESHOLD]
                    valid_neighbors = close_neighbors[close_neighbors != -1]
                    used_mask[valid_neighbors] = True

            ids_to_keep = [indices_cluster[i] for i in selected_local_indices]
            all_selected_indices.extend(ids_to_keep)

            print(f"  -> Kept {len(ids_to_keep)} / {N} images (Removed similar neighbors)")

        df_final = df.loc[all_selected_indices].copy()
        df_final = df_final.sort_values(by='difficulty', ascending=False)
        df_to_save = df_final.drop(columns=['features_vec'])

        best_perm = df_final['image_id'].tolist()
        N_final = len(best_perm)

        difficulties = df_final['difficulty'].values
        weights = np.arange(N_final, 0, -1)
        best_f1 = np.sum(difficulties * weights)

        final_features = np.stack(df_final['features_vec'].values)

        if N_final > 1:
            diffs = final_features[1:] - final_features[:-1]
            dists = np.linalg.norm(diffs, axis=1)
            pos_weights = np.arange(1, N_final)
            best_f2 = np.sum(dists / pos_weights)
        else:
            best_f2 = 0.0

        print(f"\n--- Metrics Calculated ---")
        print(f"Selected Count: {N_final}")
        print(f"Best F1 (Difficulty): {best_f1:.2f}")
        print(f"Best F2 (Diversity): {best_f2:.4f}")
        ids = df['image_id'].tolist()
        all_ids_set = set(ids)
        selected_set = set(best_perm)
        remaining = list(all_ids_set - selected_set)

        if remaining:
            rem_diffs = df.set_index('image_id').loc[remaining]['difficulty']
            remaining_sorted = rem_diffs.sort_values(ascending=False).index.tolist()
            final_elite_ids = best_perm + remaining_sorted
        else:
            final_elite_ids = best_perm

        return final_elite_ids

    @staticmethod
    def find_elbow_point(curve):
        """
        Trouve le point d'inflexion (coude) d'une courbe en cherchant le point
        le plus éloigné de la ligne droite reliant le début et la fin.
        """
        n_points = len(curve)
        all_coords = np.vstack((range(n_points), curve)).T
        first_point = all_coords[0]
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        vec_from_first = all_coords - first_point
        scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        idx_elbow = np.argmax(dist_to_line)
        return idx_elbow, curve[idx_elbow]

    @staticmethod
    def log_lighting_analysis(df):
        """
        Analyse et visualisation de l'effet de la transformation logarithmique
        sur la colonne 'lighting' du DataFrame.
        """
        plots_dir = "outputs/plots"
        os.makedirs(plots_dir, exist_ok=True)
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        sns.histplot(df['lighting'], kde=True)
        plt.title("A. Données Brutes (Échelle énorme)")
        plt.xlabel("Intensité")

        scaler = StandardScaler()
        light_scaled_only = scaler.fit_transform(df[['lighting']])
        plt.subplot(1, 3, 2)
        sns.histplot(light_scaled_only, kde=True)
        plt.title("B. StandardScaler Seul")
        plt.xlabel("Z-Score")

        light_log = np.log1p(df[['lighting']])
        light_log_scaled = scaler.fit_transform(light_log)

        plt.subplot(1, 3, 3)
        sns.histplot(light_log_scaled, kde=True, color='green')
        plt.title("C. Log + StandardScaler (Meilleur)")
        plt.xlabel("Z-Score (Log)")

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "comparaison_light_scaling.png"))
        plt.show()
