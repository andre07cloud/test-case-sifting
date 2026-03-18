import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import random
from src.common.boxplot import plot_boxplot
from datetime import datetime
import sys

file_path = "data/01-07-2024-all_tests_ga.json"
with open(file_path, "r") as f:
    data = json.load(f)


def extract_features(test_data):
    """
    Transforme un test en un vecteur exploitable pour la distance cosinus.
    On utilise les positions et rotations des objets du test.
    """
    fitness = test_data["fitness"]
    vector = []
    for obj in test_data["test"][0]: 
        position = obj[0] 
        rotation = obj[1] 
        vector.extend(position + rotation) 
    #print("Vecteur extrait :", vector)
    return np.array(vector), fitness

def test_prioritization(test_accounts):
    
    test_vectors = []
    test_ids = []
    test_objects = {}

    for run_key, test_cases in data.items():
        for test_id, test_info in test_cases.items():
            if test_info.get("test_outcome") == "FAIL": 
                test_vectors.append(extract_features(test_info))
                test_ids.append((run_key, test_id))
                test_objects[(run_key, test_id)] = test_info

    test_vectors = np.array(test_vectors)
    #print("test_accountsombre de tests extraits :", test_vectors[1:2])

    distance_matrix = cosine_distances(test_vectors)

    selected_indices = [random.randint(0, len(test_vectors) - 1)]
    print("Indice du premier test aléatoire :", selected_indices)
    while len(selected_indices) < test_accounts:
        remaining_indices = list(set(range(len(test_vectors))) - set(selected_indices))
        
        
        mean_diversity = {
            i: np.mean([distance_matrix[i, j] for j in selected_indices])
            for i in remaining_indices
        }
        #print("Diversité moyenne des tests restants :", mean_diversity)
        
        best_candidate = max(mean_diversity, key=mean_diversity.get)
        print("Meilleur candidat :", best_candidate)
        selected_indices.append(best_candidate)

    # Afficher les tests sélectionnés
    selected_tests = [test_ids[i] for i in selected_indices]
    #print("Tests sélectionnés pour maximiser la diversité :", selected_tests)

    selected_tests_data = {}
    for idx in selected_indices:
        run_key, test_id = test_ids[idx]
        if run_key not in selected_tests_data:
            selected_tests_data[run_key] = {}
        selected_tests_data[run_key][test_id] = test_objects[(run_key, test_id)]
    #print(selected_tests_data)
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_path = f"data/selected_json/{current_date}-{test_accounts}_selected_tests.json"
    with open(output_path, "w") as f:
        json.dump(selected_tests_data, f, indent=4)

    # Calculer les distances moyennes pour chaque test sélectionné (méthode optimisée)
    mean_distances_optimized = [
        np.mean([distance_matrix[i, j] for j in selected_indices if i != j])
        for i in selected_indices
    ]

    # Sélection aléatoire de nombre tests pour comparaison
    random_indices = random.sample(range(len(test_vectors)), len(selected_indices))
    mean_distances_random = [
        np.mean([distance_matrix[i, j] for j in random_indices if i != j])
        for i in random_indices
    ]

    data_plot = [mean_distances_random, mean_distances_optimized]
    plot_boxplot(data_plot, test_accounts)
    
    
if __name__ == "__main__":
    
    test_accounts = sys.argv[1]
    print("****** ARGS ********: ", test_accounts)
    test_prioritization(test_accounts=int(test_accounts))