import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.flattened_data import FeatureFlattener
from src.common.hierarchical_clustering import ClusteringEngine
from src.common.load_config_file import ConfigLoader
import sys
import time
import pickle

if __name__ == "__main__":

    config_file = "config.yaml"

    THRESHOLD = 0.50

    if len(sys.argv) > 2:
        use_case = sys.argv[2]
        data1_path = f"{use_case}_data1.pkl"
        data2_path = f"{use_case}_data2.pkl"
        # Charger les données si elles existent
        data1 = pickle.load(open(data1_path, "rb")) if os.path.exists(data1_path) else None
        data2 = pickle.load(open(data2_path, "rb")) if os.path.exists(data2_path) else None
    base_path = os.path.join("data", use_case)
    if len(sys.argv) > 3:
        metric = sys.argv[3]
        print("Usage: python main_manual_analysis.py <use_case>")

    engine = ClusteringEngine(use_case)

    if sys.argv[1] == 'flatten':
        cycle = sys.argv[3] if len(sys.argv) > 3 else 0
        flattener = FeatureFlattener(config_file, use_case)
        df, data1 = flattener.flatten(cycle=cycle)
        pickle.dump(data1, open(data1_path, "wb"))
        print("Flattened data 1 saved.")

    elif sys.argv[1] == 'cluster':
        cycle = sys.argv[3] if len(sys.argv) > 3 else 0
        data = os.path.join(base_path, f"{use_case}_cycle{cycle}_flattened_scene_features.csv")
        _, data2 = engine.kmeans_clustering(data, cycle=cycle)
        pickle.dump(data2, open(f"cycle{cycle}_{data2_path}", "wb"))
        print("Cluster data 2 saved.")

    elif sys.argv[1] == 'normal':
        mode = sys.argv[1]
        cycle = sys.argv[4] if len(sys.argv) > 4 else 0
        data = os.path.join(base_path, f"{use_case}_cycle{cycle}_features_with_clusters_kmeans.csv")
        engine.diversity_clustering(data, mode, metric, THRESHOLD, cycle)

    elif sys.argv[1] == 'inverse':
        cycle = sys.argv[4] if len(sys.argv) > 4 else 0
        loader = ConfigLoader(config_file)
        config = loader.load()
        data = config["apfd"][f"{use_case}"]["cluster"]

        if data1 is None:
            print("Generating flattened data 1...")
            flattener = FeatureFlattener(config_file, use_case)
            df, data1 = flattener.flatten(cycle=cycle)
            pickle.dump(data1, open(f"cycle{cycle}_{data1_path}", "wb"))

        if data2 is None:
            print("Generating cluster data 2...")
            _, data2 = engine.kmeans_clustering(data=data1, cycle=cycle)
            pickle.dump(data2, open(f"cycle{cycle}_{data2_path}", "wb"))
        mode = sys.argv[1]
        if mode == 'inverse':
            engine.diverse_before_clustering(data2, mode, metric, THRESHOLD)
        elif mode == 'normal':
            engine.diversity_clustering(data2, mode, metric, THRESHOLD, cycle)
