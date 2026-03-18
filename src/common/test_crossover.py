

from pymoo.core.crossover import Crossover
import numpy as np

class TestCaseCrossover(Crossover):
    def __init__(self):
        super().__init__(n_parents=2, n_offsprings=1)

    def _do(self, problem, X, **kwargs):
        """
        Perform crossover on the last axis of the input vectors.
        """
        n_matings = X.shape[1]  # Number of matings
        n_features = X.shape[2]  # Number of features in each vector
        offsprings = np.empty((self.n_offsprings, n_matings, n_features), dtype=X.dtype)

        for i in range(n_matings):
            # Extract parents
            parent1, parent2 = X[0, i], X[1, i]

            # Perform one-point crossover
            crossover_point = n_features // 2  # Split in the middle
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            # Store the child
            offsprings[0, i] = child

        return offsprings

"""
def crossover(parent1, parent2):
    Perform crossover between two parents to create a child.
    # Simple one-point crossover
    crossover_point = len(parent1) // 2
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child
"""