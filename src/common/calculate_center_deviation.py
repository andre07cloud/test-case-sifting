import math
import numpy as np


class CenterDeviationMetrics:
    """Collection of distance and center deviation metric static methods."""

    @staticmethod
    def euclidean(x, y) -> float:
        return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))

    @staticmethod
    def manhattan(x, y) -> float:
        return sum([abs(x[i] - y[i]) for i in range(len(x))])

    @staticmethod
    def minkowski(x, y, p) -> float:
        return sum([abs(x[i] - y[i]) ** p for i in range(len(x))]) ** (1 / p)

    @staticmethod
    def center_deviation(test_data, center) -> tuple:
        boxes = test_data["test"][0]
        max_deviation = -float("inf")
        deviations = []
        for i in range(len(boxes)):
            pos_i = boxes[i][0]
            deviation = CenterDeviationMetrics.euclidean(pos_i, center)
            deviations.append(deviation)
            if deviation > max_deviation:
                max_deviation = deviation
        return max_deviation, np.median(deviations), deviations
