import random
from pymoo.core.mutation import Mutation
import numpy as np

class TestCaseMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            test_case = X[i]
            # Perform a swap mutation on the test case (permutation).
            mutated_test_case = mutate(test_case)
            # Replace the original test case with the mutated one
            X[i] = mutated_test_case

        return X




def mutate(test_case):
    #Perform a swap mutation on a test case (permutation).
    mutated_test_case = test_case[:]
    # Select two random indices to swap
    idx1, idx2 = random.sample(range(len(test_case)), 2)
    # Swap the elements at the selected indices
    mutated_test_case[idx1], mutated_test_case[idx2] = mutated_test_case[idx2], mutated_test_case[idx1]

    return mutated_test_case
    


