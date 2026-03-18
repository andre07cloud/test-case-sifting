import pandas as pd
import numpy as np
import time
import sys


class APFDCalculator:
    """Calculates the Average Percentage of Faults Detected (APFD) metric."""

    def __init__(self, fault_percentile: int = 90):
        self.fault_percentile = fault_percentile

    def calculate(self, df: pd.DataFrame) -> float:
        """
        Calcule l'Average Percentage of Faults Detected (APFD).

        Args:
            df (pd.DataFrame): DataFrame containing 'image_id' and 'difficulty' columns.

        Returns:
            float: APFD score as a percentage.
        """
        df = df[['image_id', 'difficulty']].copy()
        df['image_id'] = df['image_id'].astype(int)
        N_total = len(df)
        df['ciou'] = 1 - df['difficulty']

        # Définir le seuil de difficulté pour les fautes
        fault_threshold = df['difficulty'].quantile(self.fault_percentile / 100)

        # Créer l'ensemble de fautes binaire
        df['is_fault'] = (df['difficulty'] >= fault_threshold).astype(int)

        # Dictionnaire du Fault Set pour la fonction APFD
        fault_set_map = dict(zip(df['image_id'], df['is_fault']))
        N_faults = df['is_fault'].sum()

        if N_faults == 0:
            sys.exit()

        ordered_test_ids = df['image_id'].tolist()

        n = len(ordered_test_ids)  # Nombre total de tests
        m = sum(fault_set_map.values())  # Nombre total de fautes (1s dans le dict)

        if m == 0:
            return 0.0  # Pas de faute, APFD est 0

        # Trouver la position (TF) où chaque faute est détectée pour la première fois
        faults_detected = set()
        time_to_detect_sum = 0

        for rank, test_id in enumerate(ordered_test_ids, 1):  # rank commence à 1
            if fault_set_map.get(test_id) == 1 and test_id not in faults_detected:
                time_to_detect_sum += rank
                faults_detected.add(test_id)

                if len(faults_detected) == m:
                    # Optimisation: toutes les fautes sont trouvées
                    break

        # Formule APFD: 1 - (TF_sum / (n * m)) + 1 / (2 * n)
        apfd_score = 1 - (time_to_detect_sum / (n * m)) + (1 / (2 * n))
        return apfd_score * 100  # Retourner en pourcentage
