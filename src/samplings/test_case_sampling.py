import numpy as np
import random
from pymoo.core.sampling import Sampling

#RANDOM SAMPLING
class RandomFeasibleSampling(Sampling):
    def __init__(self, test_data):
        """
        initialize random sampling, respecting the problem.
        """
        super().__init__()
        self.test_data = test_data
        self.n_tests = len(test_data)

    def _do(self, problem, n_samples, **kwargs):
        """
        Generates a matrix of solutions of form (n_samples, n_test)
        Each row is a random permutation of indices [0, ..., n_test-1].
        """
        X = np.zeros((n_samples, self.n_tests), dtype=int)  # Matrix of orderings test cases

        for i in range(n_samples):
            
            X[i] = np.random.permutation(self.n_tests)  # Random permutation of indices
        #print("X shape:", X.shape)
        #print("X:", X)
        return X


class SubsetRandomSampling(Sampling):
    def __init__(self, prob=0.05):
        super().__init__()
        self.prob = prob
        
    def _do(self, problem, n_samples, **kwargs):
        """
        Génère une matrice binaire (n_samples, n_test).
        """
        # On initialise avec une probabilité de 50% de garder chaque test
        # Astuce : Pour converger plus vite vers des petits subsets, on peut baisser p (ex: 0.3)
        #X = np.random.random((n_samples, problem.n_var)) < 0.9  # 90% de chance de garder chaque test
        X = np.random.random((n_samples, problem.n_var)) < self.prob
        return X