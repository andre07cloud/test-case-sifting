import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.performance_analysis import PerformanceAnalyzer
import sys


if __name__ == "__main__":
    config_file = "config.yaml"
    if len(sys.argv) > 2:
        use_case = sys.argv[1]
        mode = sys.argv[2]
        cycle = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        print("Usage: python main_performance.py <use_case> <test_bucket>")
        exit("Please provide a use case argument. uc1 or uc2 and test_bucket.")

    analyzer = PerformanceAnalyzer(config_file, use_case)
    analyzer.evaluate_per_bucket(mode, cycle)
