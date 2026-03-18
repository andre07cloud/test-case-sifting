import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.gain_calculate import GainCalculator
import sys
import os

if __name__ == "__main__":

    use_case = sys.argv[1]
    cycle_start = 0
    cycle_end = 2
    base_path = os.path.join("data", use_case, f"normal_{use_case}_test_manual_analysis")
    difficulty_stats_path = os.path.join("data", use_case)

    calculator = GainCalculator(use_case)

    for i in range(cycle_start, cycle_end):
        print(f"************* Evaluation over the cycle {i} to {i+1} *************")

        file_baseline = os.path.join(base_path, f"cycle{i}", f'cycle{i}_{use_case}_normal_evaluation_rq1_summary.csv')
        file_finetuned = os.path.join(base_path, f"cycle{i+1}", f'cycle{i+1}_{use_case}_normal_evaluation_rq1_summary.csv')

        calculator.calculate_gains(file_baseline, file_finetuned, target_col='Overall_elite', cycle_start=i, cycle_end=i+1)

        stats_baseline = os.path.join(difficulty_stats_path, f"{use_case}_cycle{i}_kmeans_difficulty_stats.json")
        stats_end = os.path.join(difficulty_stats_path, f"{use_case}_cycle{i+1}_kmeans_difficulty_stats.json")
        calculator.calculate_reduction_factor(stats_baseline, stats_end, cycle_start=i, cycle_end=i+1)

    file_baseline = os.path.join(base_path, f"cycle{cycle_start}", f'cycle{cycle_start}_{use_case}_normal_evaluation_rq1_summary.csv')
    file_finetuned = os.path.join(base_path, f"cycle{cycle_end}", f'cycle{cycle_end}_{use_case}_normal_evaluation_rq1_summary.csv')
    calculator.calculate_gains(file_baseline, file_finetuned, target_col='Overall_elite', cycle_start=cycle_start, cycle_end=cycle_end)

    stats_baseline = os.path.join(difficulty_stats_path, f"{use_case}_cycle{cycle_start}_kmeans_difficulty_stats.json")
    stats_end = os.path.join(difficulty_stats_path, f"{use_case}_cycle{cycle_end}_kmeans_difficulty_stats.json")
    calculator.calculate_reduction_factor(stats_baseline, stats_end, cycle_start=cycle_start, cycle_end=cycle_end)

    calculator.plot_difficulty_and_apfd(cycle_start=cycle_start, cycle_end=cycle_end)
