import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.manual_analysis import ManualAnalyzer
import sys

if __name__ == "__main__":
    config_file = "config.yaml"
    if len(sys.argv) > 2:
        use_case = sys.argv[1]
        mode = sys.argv[2]
        cycle = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        print("Usage: python main_manual_analysis.py <use_case>")
        exit("Please provide a use case argument. uc1 or uc2")
    analyzer = ManualAnalyzer(config_file, use_case)
    analyzer.analyze(mode, int(cycle))
