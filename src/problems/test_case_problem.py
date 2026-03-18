
from pymoo.core.problem import Problem
import numpy as np


# --- 3. Définition du Problème Mono-Objectif Pondéré ---
class WeightedScoreProblem(Problem):
    def __init__(self, conflict_matrix, difficulties, alpha=1.0, beta=100.0):
        self.conflict_matrix = conflict_matrix
        self.difficulties = difficulties
        self.alpha = alpha # Poids de la Difficulté
        self.beta = beta   # Poids de la Diversité (Pénalité de conflit)
        self.n_items = len(difficulties)
        
        # n_obj=1 : On maximise un seul score
        # n_constr=0 : La contrainte est gérée directement dans la fitness (pénalité)
        super().__init__(n_var=self.n_items, n_obj=1, n_constr=0, xl=0, xu=1, type_var=bool)

    def _evaluate(self, x, out, *args, **kwargs):
        f = []
        
        for ind in x:
            # 1. Calculer la Difficulté Totale (Maximiser)
            # Somme des difficultés des images sélectionnées
            total_diff = np.sum(self.difficulties[ind])
            
            # 2. Calculer les Conflits (Minimiser)
            # Nombre de paires trop proches
            ind_int = ind.astype(int)
            # (x^T * C * x) / 2
            # Optimisation: Si size < 2, conflicts = 0
            if np.sum(ind_int) < 2:
                n_conflicts = 0
            else:
                n_conflicts = (ind_int @ self.conflict_matrix @ ind_int) / 2
            
            # 3. La Formule Magique (Fitness)
            # Score = (alpha * Diff) - (beta * Conflits)
            # Pymoo MINIMISE, donc on retourne l'opposé : -(Score)
            
            # Note: Ici, "Diversity Score" est interprété comme "Absence de Conflits".
            # Si Beta est haut, chaque conflit coûte très cher, donc l'algo va les éliminer.
            fitness = (self.alpha * total_diff) - (self.beta * n_conflicts)
            
            f.append(-fitness) # On minimise le négatif pour maximiser le positif

        out["F"] = np.array(f)


class TestCaseReductionProblem(Problem):
    
    def __init__(self, conflict_matrix, difficulties):
        """
        Initialise le problème de RÉDUCTION multi-objectifs.
        
        Args:
            conflict_matrix (np.array): Matrice NxN (1=conflit, 0=ok).
            difficulties (np.array): Vecteur de taille N avec les difficultés.
        """
        self.conflict_matrix = conflict_matrix
        self.difficulties = difficulties
        self.n_tests = len(difficulties)

        # --- CHANGEMENT MAJEUR ---
        # n_var = Nombre de tests
        # type_var = bool (Binaire : 1=Gardé, 0=Jeté)
        # n_obj = 2 (Minimiser Taille, Maximiser Difficulté)
        # n_constr = 1 (Contrainte de Diversité : 0 Conflit)
        super().__init__(n_var=self.n_tests, n_obj=2, n_constr=0, xl=0, xu=1, type_var=bool)
        
    def _evaluate(self, X, out, *args, **kwargs):
        """
        X est une matrice (Population_Size x N_Tests) de booléens.
        """
        #print("Population :", X)
        f1_list = [] # Taille
        f2_list = [] # Difficulté (Négative)
        #g1_list = [] # Contrainte Conflits
        
        # Pour optimiser la vitesse, on évite les boucles python si possible
        # Mais pour la clarté ici, je garde une boucle par individu
        PENALTY = 5.0
        for ind in X:
            #print("Individu:", ind)
            # ind est un vecteur booléen [True, False, True...]
            
            # --- OBJECTIF 1 : Minimiser la Taille ---
            # Somme des bits à 1
            size = np.sum(ind)

            
            # --- OBJECTIF 2 : Maximiser la Difficulté Totale ---
            # On veut maximiser, or Pymoo minimise toujours.
            # Donc on calcule la somme et on prend l'opposé (-).
            # On utilise le masque 'ind' pour sommer seulement les difficultés sélectionnées.
            total_diff = np.sum(self.difficulties[ind])
            f2_list.append(-total_diff)
            
            # --- CONTRAINTE : Diversité (0 Conflit) ---
            # Calcul rapide du nombre de conflits dans le sous-graphe sélectionné
            # Formule algébrique : n_conflicts = (x^T * C * x) / 2
            # On convertit en int (0/1) pour le calcul matriciel
            ind_int = ind.astype(int)
            
            # On ne regarde que la sous-matrice des éléments sélectionnés
            # Si taille < 2, pas de conflit possible
            if size < 2:
                n_conflicts = 0
            else:
                # Produit matriciel intelligent : 
                # On projette le vecteur sur la matrice de conflit
                # Si ind[i]=1 et ind[j]=1 et C[i,j]=1, ça ajoute au score
                n_conflicts = (ind_int @ self.conflict_matrix @ ind_int) / 2
            
            score = -size + (PENALTY * n_conflicts)
            f1_list.append(score)
            # Pymoo considère la contrainte violée si G > 0.
            # On veut n_conflicts <= 0 (donc exactement 0).
            #g1_list.append(n_conflicts)
        # print("F1 (Sizes):", f1_list)
        # print("F2 (Neg Diffs):", f2_list)
        # print("G1 (Conflicts):", g1_list)
        # Assignation des résultats
        out["F"] = np.column_stack([f1_list, f2_list])
        #print(out["F"])
        #out["G"] = np.column_stack([g1_list])
        #print(out["G"])


class TestCasePrioritizationProblem(Problem):
    
    def __init__(self, all_test_ids, data_map):
        """
        Initialise le problème de priorisation multi-objectifs.

        Args:
            all_test_ids (list): Liste des IDs réels.
            data_map (dict): Dictionnaire de référence {ID: {difficulty, features_scaled}}.
        """
        
        self.data_map = data_map
        self.all_test_ids = all_test_ids
        
        n_test = len(all_test_ids)

        # Chaque solution est une permutation des indices [0 à n_test-1]
        xl = np.zeros(n_test, dtype=int)
        xu = np.array([n_test - 1] * n_test, dtype=int)
        
        # --- MODIFICATION ICI ---
        # n_obj=2 : On active deux objectifs (Difficulté et Diversité)
        super().__init__(n_var=n_test, n_obj=2, n_constr=0, xl=xl, xu=xu)
        
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Évalue la population.
        f1: Difficulté pondérée (Maximisation)
        f2: Diversité Séquentielle Pondérée (Formule de l'article)
        """
        
        N = self.n_var
        f1_list = []
        f2_list = []
        
        # Boucle sur chaque candidat (permutation)
        for candidate_indices in X:
            
            # 1. Récupération des IDs et des Features dans l'ordre de la permutation
            candidate_ids = [self.all_test_ids[i] for i in candidate_indices]
            
            # On crée une matrice (N x M) contenant toutes les features ordonnées
            # C'est beaucoup plus rapide que de faire des boucles for
            features_sequence = np.array([self.data_map[tid]['features_scaled'] for tid in candidate_ids])
            
            # ---------------------------------------------------------
            # OBJECTIF 1 : Difficulté Pondérée (APFD-like logic)
            # ---------------------------------------------------------
            difficulties = np.array([self.data_map[tid]['difficulty'] for tid in candidate_ids])
            # Poids décroissants : [N, N-1, N-2, ..., 1]
            weights_difficulty = np.arange(N, 0, -1)
            sum_weighted_difficulty = np.sum(difficulties * weights_difficulty)

# --- OBJ 2: Diversité Séquentielle (AVEC ZERO-PADDING) ---
            
            # 1. Récupérer les vecteurs bruts (qui peuvent avoir des tailles différentes)
            # On s'assure que ce sont des arrays numpy, sinon on convertit
            raw_features = [np.array(self.data_map[tid]['features_scaled']) for tid in candidate_ids]
            
            # 2. Trouver la dimension maximale parmi tous les tests de cette séquence
            # Si la liste est vide (cas improbable), on met 0
            if raw_features:
                max_len = max(len(f) for f in raw_features)
            else:
                max_len = 0

            # 3. Appliquer le Zero-Padding
            # Si un vecteur est plus petit que max_len, on comble la différence avec des 0
            padded_features_list = []
            for f in raw_features:
                pad_width = max_len - len(f)
                if pad_width > 0:
                    # Ajoute 'pad_width' zéros à la fin du vecteur
                    f_padded = np.pad(f, (0, pad_width), 'constant')
                    padded_features_list.append(f_padded)
                else:
                    padded_features_list.append(f)
            
            # Maintenant, features_sequence est une matrice rectangulaire parfaite (N x max_len)
            features_sequence = np.array(padded_features_list)
            
            # 4. Calcul de la diversité (Code standard vectorisé)
            # t_i - t_{i-1}
            diffs = features_sequence[1:] - features_sequence[:-1]
            consecutive_distances = np.linalg.norm(diffs, axis=1)
            
            # Pondération par la position (1/i)
            if N > 1:
                position_weights = np.arange(1, N)
                weighted_distances = consecutive_distances / position_weights
                diversity_score = np.sum(weighted_distances)
            else:
                diversity_score = 0.0
            # ---------------------------------------------------------
            # Sorties Pymoo (Minimisation)
            # ---------------------------------------------------------
            fitness1 = -sum_weighted_difficulty/10000
            fitness2 = -diversity_score
            f1_list.append(fitness1) # On maximise la difficulté
            f2_list.append(fitness2)         # On maximise la diversité
            
        out["F"] = np.column_stack([f1_list, f2_list])
    
    # def _evaluate(self, X, out, *args, **kwargs):
    #     """
    #     Évalue une population de permutations (X).
    #     f1: Difficulté pondérée (Maximisation -> Minimisation via négation)
    #     f2: Diversité incrémentale (Maximisation -> Minimisation via négation)
    #     """
        
    #     N = self.n_var
    #     f1_list = []
    #     f2_list = []
        
    #     # Boucle sur chaque candidat (permutation) dans la population
    #     for candidate_indices in X:
            
    #         # Conversion des indices Pymoo en IDs réels
    #         candidate_ids = [self.all_test_ids[i] for i in candidate_indices]

    #         sum_weighted_difficulty = 0.0
    #         sum_incremental_diversity = 0.0
    #         executed_features_list = []

    #         for rank, test_id in enumerate(candidate_ids):
                
    #             # Récupération des données
    #             data = self.data_map.get(test_id)
    #             difficulty = data['difficulty']
    #             current_features = data['features_scaled']
                
    #             # --- Objectif 1: Difficulté Pondérée ---
    #             # On veut exécuter les tests difficiles en premier (rank petit -> (N-rank) grand)
    #             sum_weighted_difficulty += difficulty * (N - rank)
                
    #             # --- Objectif 2: Diversité Incrémentale ---
    #             if executed_features_list:
    #                 # Conversion en matrice numpy pour calcul vectorisé
    #                 previous_features_matrix = np.vstack(executed_features_list)
                    
    #                 # Distance Euclidienne entre le test courant et tous les précédents
    #                 distances = np.linalg.norm(previous_features_matrix - current_features, axis=1)
                    
    #                 # Pour maximiser la couverture, on veut être le plus loin possible 
    #                 # du test le plus proche déjà exécuté (stratégie Maximin)
    #                 contribution = np.min(distances)
    #                 sum_incremental_diversity += contribution
    #             else:
    #                 # Le premier test n'a pas de prédécesseur, contribution 0 (ou valeur par défaut)
    #                 sum_incremental_diversity += 0.0
                
    #             # Ajouter aux features exécutées
    #             executed_features_list.append(current_features)
            
    #         # --- MODIFICATION ICI ---
    #         # Pymoo minimise toujours. On passe les valeurs en négatif pour maximiser.
    #         f1_list.append(-sum_weighted_difficulty)
    #         f2_list.append(-sum_incremental_diversity)
            
    #     # --- MODIFICATION ICI ---
    #     # On retourne une matrice numpy avec 2 colonnes [f1, f2]
    #     out["F"] = np.column_stack([f1_list, f2_list])

# class TestCasePrioritizationProblem(Problem):
#     def __init__(self, test_cases_data):
#         # n_var : umber of variables describing a test case
        
#         self.test_cases_data = test_cases_data  # List of all test cases
#         n_test = len(test_cases_data)
#         # Each solution is a permutation of indices from 0 to n_test-1
#         xl = np.zeros(n_test)
#         xu = np.array([n_test - 1] * n_test)
#         super().__init__(n_var=n_test, n_obj=2, n_constr=0, xl=xl, xu=xu)
        

#     def _evaluate(self, X, out, *args, **kwargs):
 
#         f1_list, f2_list = [], []
#         for candidate in X:
            
#             candidate_order = [self.test_cases_data[i] for i in candidate]
#             #print("Candidate order len:", len(candidate_order))
#             #print("Candidate:", candidate)
#             #vision_complexity = - fitness_vison_complexity(candidate_order)
#             test_fitness = [item[1] for item in candidate_order]
#             fit1 = np.mean(test_fitness)
#             #print("Test fitness:", test_fitness)
#             #fit1 = test_fitness
#             diversity = - compute_diversity(candidate_order)
#             f1_list.append(fit1)
#             f2_list.append(diversity)
            
#         #fit3 = -np.mean(f1_list)
#         out["F"] = np.column_stack([np.array(f1_list), np.array(f2_list)])
