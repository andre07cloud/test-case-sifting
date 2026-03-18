import matplotlib.pyplot as plt
import os


class Visualizer:
    """Generates visualization plots for test case prioritization results."""

    def __init__(self, plots_dir: str = "outputs/plots"):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_boxplot(self, data, num_tests):
        output_path = self.plots_dir
        # Create the box plot
        plt.figure(figsize=(10, 8))
        plt.boxplot(data, labels=["Random selection", "Optimized selection"])
        plt.title(f'Box Plot of {num_tests} Test Case Prioritization')
        plt.ylabel('Average Diversity')
        plt.xlabel('Selector')
        plt.grid(True)

        # Show the plots
        plt.tight_layout()
        plt.savefig(f'{output_path}/{num_tests}_av_diversity_boxplot.png')
        plt.show()
