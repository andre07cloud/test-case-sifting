import math
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class DistanceMetrics:
    """Collection of distance metric static methods and utility functions."""

    @staticmethod
    def euclidean(x, y) -> float:
        return euclidean_distances(x, y)

    @staticmethod
    def manhattan(x, y) -> float:
        return sum([abs(x[i] - y[i]) for i in range(len(x))])

    @staticmethod
    def minkowski(x, y, p) -> float:
        return sum([abs(x[i] - y[i]) ** p for i in range(len(x))]) ** (1 / p)

    @staticmethod
    def min_distance(test_data) -> tuple:
        boxes = test_data["test"][0]
        min_distance = float("inf")
        distances = []
        for i in range(len(boxes)):
            pos_i = boxes[i][0]
            for j in range(i + 1, len(boxes)):
                pos_j = boxes[j][0]
                distance = DistanceMetrics.euclidean(pos_i, pos_j)
                distances.append(distance)
                if distance < min_distance:
                    min_distance = distance
        return min_distance, np.median(distances), distances
