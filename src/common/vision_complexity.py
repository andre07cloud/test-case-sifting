from .calculate_center_deviation import CenterDeviationMetrics
from .calculate_max_rotation import RotationMetrics
from .calculate_distance import DistanceMetrics


class VisionComplexityEvaluator:
    """Evaluates vision complexity of a test scene using distance, deviation and rotation metrics."""

    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def evaluate(self, test_data) -> float:
        center = [-0.7, 0.26, 0.786]
        reference_angle = 90
        max_center_distance, _, _ = CenterDeviationMetrics.center_deviation(test_data, center)
        min_distance, _, _ = DistanceMetrics.min_distance(test_data)
        max_deviation, _, _ = RotationMetrics.rotation_deviation(test_data, reference_angle)

        print("Max Center Distance: ", max_center_distance)
        print("Min Distance: ", min_distance)
        print("Max Deviation: ", max_deviation)
        fitness_vision = self.alpha * max_center_distance + self.beta * min_distance + self.gamma * max_deviation
        print("Fitness Vision:", fitness_vision)
        return fitness_vision
