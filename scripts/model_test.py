import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from src.common.perception_difficulty import PerceptionDifficultyEvaluator
import os

if __name__ == "__main__":

    model = YOLO("/home/andredejesus/My-PhD/Sycodal/test-case-prioritization/train_results/uc1/train_results_uc12/weights/best.pt")
    # Test the model
    image_path = "/home/andredejesus/My-PhD/Sycodal/uc1_26_nov_4woods_uniq/images/13.png"
    results = model(image_path, show=True, save=True)  # predict on an image
    print(f"list of results: {results[0].masks} and type: {type(results)}")
    print("Model testing completed.")

    config_file = "config.yaml"
    use_case = "uc1"
    label_path = None
    evaluator = PerceptionDifficultyEvaluator(config_file, use_case)
    diff = evaluator.evaluate_detection(image_path, label_path)


#+++++++++++++++++++++
# EVALUATE PERCEPTION DIFFICULTY
#+++++++++++++++++++++
