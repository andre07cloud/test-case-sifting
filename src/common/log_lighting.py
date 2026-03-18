import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Simulation de votre cas (valeurs de lumière très étendues)
# Supposons que 'df' est votre DataFrame et 'lighting' la colonne
# df = pd.read_csv('votre_fichier.csv') 

# --- 1. Visualisation du Problème ---

def log_lighting_analysis(df):
    """
    Analyse et visualisation de l'effet de la transformation logarithmique
    sur la colonne 'lighting' du DataFrame.
    """
    plt.figure(figsize=(15, 5))

    # Cas A : Données Brutes
    plt.subplot(1, 3, 1)
    sns.histplot(df['lighting'], kde=True)
    plt.title("A. Données Brutes (Échelle énorme)")
    plt.xlabel("Intensité")

    # Cas B : StandardScaler seul
    scaler = StandardScaler()
    light_scaled_only = scaler.fit_transform(df[['lighting']])
    plt.subplot(1, 3, 2)
    sns.histplot(light_scaled_only, kde=True)
    plt.title("B. StandardScaler Seul")
    plt.xlabel("Z-Score")

    # Cas C : Log + StandardScaler (RECOMMANDÉ)
    # np.log1p calcule log(1 + x), utile si vous avez des 0
    light_log = np.log1p(df[['lighting']]) 
    light_log_scaled = scaler.fit_transform(light_log)

    plt.subplot(1, 3, 3)
    sns.histplot(light_log_scaled, kde=True, color='green')
    plt.title("C. Log + StandardScaler (Meilleur)")
    plt.xlabel("Z-Score (Log)")

    plt.tight_layout()
    plt.savefig("comparaison_light_scaling.png")
    plt.show()