from ..problems.test_case_problem import *
from ..samplings.test_case_sampling import SubsetRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
import pandas as pd
import numpy as np
import sys


class GeneticAlgorithmSolver:
    """Solves test case selection using single- and multi-objective genetic algorithms."""

    def __init__(self, n_experiments=10, pop_size=100, n_gen=200, alpha=0.5, beta=0.5):
        self.n_experiments = n_experiments
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.alpha = alpha
        self.beta = beta

    def solve_single_objective(self, conflict_matrix, difficulties) -> tuple:
        ALPHA = self.alpha
        BETA = 1 - ALPHA

        problem = WeightedScoreProblem(conflict_matrix, difficulties, alpha=ALPHA, beta=BETA)
        sampling = SubsetRandomSampling(prob=0.1)
        algorithm = GA(
            pop_size=self.pop_size,
            sampling=sampling,
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(prob=0.01),
            eliminate_duplicates=True
        )

        print(f"Lancement AG Mono-Objectif (Alpha={ALPHA}, Beta={BETA})...")
        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", self.n_gen),
            seed=42,
            verbose=True
        )

        best_mask = res.X
        best_score = -res.F[0]

        best_size = np.sum(best_mask)
        idx = np.where(best_mask)[0]

        real_diff = np.sum(difficulties[best_mask])
        ind_int = best_mask.astype(int)
        real_conflicts = (ind_int @ conflict_matrix @ ind_int) / 2
        reduction_rate = 100 * (1 - best_size / problem.n_var)
        print(f"\n--- RÉSULTAT FINAL ---")
        print(f"Fitness Finale : {best_score:.4f}")
        print(f"Images retenues : {int(best_size)}")
        print(f"Difficulté Totale : {real_diff:.4f}")
        print(f"Conflits Restants : {int(real_conflicts)}")
        print(f"Réduction : {100*(1 - best_size/problem.n_var):.2f}%")

        if real_conflicts == 0:
            print("Succès : Dataset parfaitement nettoyé (0 doublon).")
        else:
            print("Attention : Il reste des conflits. Augmentez BETA (la pénalité).")

        return best_mask, idx, reduction_rate, real_diff, 0, best_size

    def solve_multi_objective(self, conflict_matrix, difficulties) -> tuple:
        problem = TestCaseReductionProblem(conflict_matrix, difficulties)
        sampling = SubsetRandomSampling(prob=0.1)
        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=sampling,
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True
        )

        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", self.n_gen),
            seed=42,
            verbose=True
        )
        print("\n--- Résultats de l'Optimisation MOGA ---")
        if res.X is None:
            print("ERREUR CRITIQUE : L'algorithme n'a trouvé aucune solution valide.")
            print("Conseil : Augmentez N_GEN, POP_SIZE ou réduisez la densité du sampling.")
            return None, [], 0, 0, 0, 0

        valid_solutions = []
        best_conflits = np.inf
        for i, solution_mask in enumerate(res.X):
            size = np.sum(solution_mask)

            if size < 2:
                n_conflicts = 0
            else:
                ind_int = solution_mask.astype(int)
                n_conflicts = (ind_int @ conflict_matrix @ ind_int) / 2

            if n_conflicts <= best_conflits:
                best_conflits = n_conflicts
                total_diff = -res.F[i, 1]
                ratio = total_diff / size if size > 0 else 0

                valid_solutions.append({
                    "mask": solution_mask,
                    "size": size,
                    "difficulty": total_diff,
                    "ratio": ratio,
                    "conflicts": best_conflits
                })

        if len(valid_solutions) > 0:
            print(f"\nNombre de solutions valides (0 conflit) trouvées : {len(valid_solutions)}")

            best_sol_data = max(valid_solutions, key=lambda x: x["ratio"])

            best_solution = best_sol_data["mask"]
            best_size = best_sol_data["size"]
            best_diff = best_sol_data["difficulty"]

            reduction_rate = 100 * (1 - best_size / problem.n_var)
            print(f"best soltion: {best_solution}")
            indices_finaux = np.where(best_solution)[0]

            print(f"--- Meilleure Solution Exp ---")
            print(f"Images retenues : {int(best_size)}")
            print(f"Difficulté Totale : {best_diff:.2f}")
            print(f"Réduction : {reduction_rate:.2f}%")

            return best_solution, indices_finaux, reduction_rate, best_diff, 0, best_size

        else:
            print("ERREUR : Aucune solution sans conflit (0 doublon) n'a été trouvée.")
            print("Conseil : Augmentez la PENALTY dans le Problem ou le nombre de générations.")
            return None, [], 0, 0, 0, 0

    def random_search(self, n_iterations, data, dist_matrix, difficulties, threshold) -> list:
        """
        Effectue N tris aléatoires et applique le filtre glouton.
        Retourne la solution qui a le meilleur ratio Difficulté/Taille.
        """
        best_indices = []
        best_score = -1

        for _ in range(n_iterations):
            shuffled_indices = np.random.permutation(len(data))

            kept_indices = []
            for idx in shuffled_indices:
                if not kept_indices:
                    kept_indices.append(idx)
                else:
                    dists = dist_matrix[idx, kept_indices]
                    if np.min(dists) > threshold:
                        kept_indices.append(idx)

            size = len(kept_indices)
            total_diff = np.sum(difficulties[kept_indices])
            score = total_diff / size

            if score > best_score:
                best_score = score
                best_indices = kept_indices

        return best_indices
