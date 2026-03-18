import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.data_merging import DataMerger
import sys

if __name__ == '__main__':
    config_file = "config.yaml"

    if len(sys.argv) > 1:
        use_case = sys.argv[1]
        cycle = sys.argv[2]
    else:
        print("Usage: python main_performance.py <use_case> <test_bucket>")
        exit("Please provide a use case argument. uc1 or uc2 and test_bucket.")
    dir_count = 2
    merger = DataMerger(config_file, use_case)
    merger.merge(dir_count, cycle)
