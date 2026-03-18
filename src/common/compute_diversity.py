import numpy as np
from .data_extraction import DataExtractor
from .calculate_distance import DistanceMetrics
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import sys
from .load_config_file import ConfigLoader


class DiversityCalculator:
    """Calculates diversity metrics for test case ordering (MOGA fitness)."""

    def __init__(self, config_file: str, use_case: str):
        self.config_file = config_file
        self.use_case = use_case
        self.all_test_ids = None
        self.data_map = None

    def preprocess(self) -> tuple:
        """
        Load and preprocess test data for the MOGA.
        Sets self.all_test_ids and self.data_map.
        Returns (all_test_ids, data_map).
        """
        config = ConfigLoader(self.config_file).load()
        uc_moga = config['moga'][f'{self.use_case}']
        FILE_PATH = uc_moga['initial_pop']

        try:
            df_full = pd.read_csv(FILE_PATH)
        except FileNotFoundError:
            print(f"ERREUR : Fichier {FILE_PATH} introuvable.")
            sys.exit()

        FEATURE_COLS = [col for col in df_full.columns if col.startswith('pos_') or col.startswith('rot_')]

        N = len(df_full)
        print(f"Total of test cases (N) : {N}")

        df_full[FEATURE_COLS] = df_full[FEATURE_COLS].fillna(0.0)
        scaler = StandardScaler()
        features_scaled_np = scaler.fit_transform(df_full[FEATURE_COLS].values)

        self.data_map = {
            row['image_id']: {
                'difficulty': row['difficulty'],
                'features_scaled': features_scaled_np[i]
            }
            for i, row in df_full.iterrows()
        }
        self.all_test_ids = df_full['image_id'].tolist()
        print("Données prêtes pour le MOGA.")
        return self.all_test_ids, self.data_map

    def compute(self, candidate_order: list) -> float:
        """
        Calculates the fitness of a candidate solution (a permutation of test cases)
        using the euclidean distance between the features of consecutive test cases.
        """
        n = len(candidate_order)
        total = 0.0
        for i in range(1, n):
            test_vector = [item[0] for item in candidate_order]
            test_i = test_vector[i]
            test_i_1 = test_vector[i - 1]

            distance = DistanceMetrics.euclidean(test_i, test_i_1)
            total += distance / i
        return total

    def calculate_moga_fitness(self, permutation_ids: list) -> tuple:
        """
        Calcule les deux scores de fitness pour une permutation donnée.

        Args:
            permutation_ids (list): Un ordre de test (e.g., [123, 45, 800, ...]).

        Returns:
            tuple: (Score_Difficulté, Score_Diversité). Les deux sont à maximiser.
        """
        if self.data_map is None:
            raise RuntimeError("Call preprocess() before calculate_moga_fitness().")

        N = len(permutation_ids)
        sum_difficulty_score = 0
        sum_diversity_score = 0

        executed_features_list = []

        for rank, test_id in enumerate(permutation_ids):
            data = self.data_map.get(test_id)
            if not data:
                continue

            difficulty = data['difficulty']
            current_features = data['features_scaled']

            sum_difficulty_score += difficulty * (N - rank)

            if executed_features_list:
                previous_features_matrix = np.vstack(executed_features_list)
                distances = np.linalg.norm(previous_features_matrix - current_features, axis=1)
                contribution = np.min(distances)
                sum_diversity_score += contribution

            executed_features_list.append(current_features)

        return (sum_difficulty_score, sum_diversity_score)
