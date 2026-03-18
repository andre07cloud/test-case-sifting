import numpy as np


class DataExtractor:
    """Static methods for extracting features from test case data."""

    @staticmethod
    def extract_features(test_data) -> tuple:
        fitness = test_data["fitness"]
        vector = []
        for obj in test_data["test"][0]:
            position = obj[0]
            rotation = obj[1]
            vector.extend(position + rotation)
        return vector, fitness

    @staticmethod
    def extract_data(data) -> tuple:
        test_vectors = []
        test_ids = []
        test_objects = {}

        for run_key, test_cases in data.items():
            for test_id, test_info in test_cases.items():
                if test_info.get("test_outcome") == "FAIL":
                    vector, fitness = DataExtractor.extract_features(test_info)
                    test_vectors.append((vector, fitness))
                    test_ids.append((run_key, test_id))
                    test_objects[(run_key, test_id)] = test_info

        return test_vectors, test_ids, test_objects
