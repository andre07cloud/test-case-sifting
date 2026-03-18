# Risk-Aware, Novelty-First Test Sifting for Continuous Evaluation of Manipulator Vision Models

A research framework for **for cost-bounded, continual evaluation and repair of manipulator vision models.** using perception difficulty, clustering, and ANN. The system iteratively refines a YOLO segmentation model by identifying hard test cases, selecting the most informative ones, and fine-tuning the model across successive cycles.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Data Structure](#data-structure)
6. [Pipeline Workflow](#pipeline-workflow)
7. [Script Reference](#script-reference)
8. [Outputs](#outputs)
9. [Reproducibility Notes](#reproducibility-notes)

---

## Overview

The framework addresses two use cases (`uc1`: wood bin picking, `uc2`: box picking) and runs iterative **cycles** of:

1. Flatten raw scene features → CSV
2. K-means clustering by difficulty
3. Approximative Nearest Neighbor for test sifting (diversity)
4. Manual analysis → Elite / Rejected splits
5. Fine-tune YOLO on hard cases
6. Evaluate and compute gains

**Key metrics:** APFD (Average Percentage of Faults Detected), mAP50-95, Precision, Recall, F1.

---

## Project Structure

```
test-case-sifting/
├── config.yaml                  # Central configuration (paths, thresholds)
├── requirements.txt
├── scripts/                     # Entry-point scripts (all executable)
│   ├── main_clustering.py
│   ├── main_data_merging.py
│   ├── main_fine_tuning_perception.py
│   ├── main_ga.py
│   ├── main_gain_over_cycle.py
│   ├── main_gain_over_cycle_withplot.py
│   ├── main_manual_analysis.py
│   ├── main_calculate_apfd.py
│   ├── main_performance.py
│   ├── main_split_for_test.py
│   ├── main_split_merge.py
│   ├── main_train_perception.py
│   ├── main_vision_complexity.py
│   ├── diversity_performance.py
│   ├── naive_cluster_performance.py
│   ├── model_test.py
│   └── test-prioritization.py
├── src/
│   ├── common/                  # Core OOP classes
│   │   ├── load_config_file.py      # ConfigLoader
│   │   ├── hierarchical_clustering.py # ClusteringEngine
│   │   ├── perception_difficulty.py  # PerceptionDifficultyEvaluator
│   │   ├── ga_algorithm.py           # GeneticAlgorithmSolver
│   │   ├── manual_analysis.py        # ManualAnalyzer
│   │   ├── performance_analysis.py   # PerformanceAnalyzer
│   │   ├── gain_calculate.py         # GainCalculator
│   │   ├── data_merging.py           # DataMerger
│   │   ├── flattened_data.py         # FeatureFlattener
│   │   ├── fine_tuning.py            # FineTuner
│   │   ├── compute_diversity.py      # DiversityCalculator
│   │   ├── calculate_apfd.py         # APFDCalculator
│   │   ├── split_range_data.py       # DatasetSplitter
│   │   ├── split_data_from_merge.py  # MergedDatasetSplitter
│   │   ├── vision_complexity.py      # VisionComplexityEvaluator
│   │   ├── data_extraction.py        # DataExtractor
│   │   ├── utils.py                  # GeometryUtils
│   │   └── boxplot.py                # Visualizer
│   ├── problems/
│   │   └── test_case_problem.py      # Pymoo problem definitions
│   └── samplings/
│       └── test_case_sampling.py     # Pymoo sampling strategies
├── data/
│   ├── uc1/                     # Use case 1 datasets (images, labels, masks)
│   └── uc2/                     # Use case 2 datasets
├── models/                      # Fine-tuned YOLO weights per cycle
├── train_results/               # YOLO training logs and metrics
├── outputs/
│   └── plots/                   # Generated figures (PNG)
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA-compatible GPU recommended (for YOLO training/inference)

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd test-case-sifting

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the base YOLO model (already included if present)
# yolo11n-seg.pt must be at the project root
# Download from: https://github.com/ultralytics/assets/releases
```

> **Note on FAISS:** If `faiss-cpu` fails on your platform, try `pip install faiss-gpu` for GPU builds, or install via conda: `conda install -c pytorch faiss-cpu`.

---

## Configuration

All paths and parameters are centralised in **`config.yaml`** at the project root.
**All paths are relative to the project root** — no absolute paths, no machine-specific edits required.

### Design principle

`ConfigLoader` (in `src/common/load_config_file.py`) resolves every path relative to the directory that contains `config.yaml`. You can also resolve paths explicitly in your own code:

```python
from src.common.load_config_file import ConfigLoader

loader = ConfigLoader("config.yaml")
cfg    = loader.config                           # dict
model  = loader.resolve(cfg["best_model"]["uc1"])  # → absolute path at runtime
```

### The only section you must update: `merging.base_path`

Place your raw simulation data (Isaac Sim output) under `data/raw/` and set:

```yaml
merging:
  uc1:
    base_path: "data/raw/uc1"   # must contain dir_1/, dir_2/, … with *_stats.json
  uc2:
    base_path: "data/raw/uc2"
```

Everything else (merged datasets, training outputs, CSVs) is written to `data/` by the pipeline scripts and requires no manual path configuration.

### Config sections overview

| Section | Used by | Description |
|---------|---------|-------------|
| `merging` | `DataMerger`, `FeatureFlattener`, `PerceptionDifficultyEvaluator` | Raw data sources and merged output destinations |
| `best_model` | `PerceptionDifficultyEvaluator`, `PerformanceAnalyzer` | Baseline YOLO weights (cycle 0) |
| `test_images` / `test_labels` | `PerceptionDifficultyEvaluator` | Evaluation dataset (detection mode) |
| `difficulty_levels` | `ManualAnalyzer` | ANN diversity CSVs per level and mode |
| `master_json_path` | `ManualAnalyzer` | Base path for merged-cycle JSON files |
| `collected_file_folder` | `ManualAnalyzer` | Base path for merged-cycle data folders |
| `main_output_folder` | `ManualAnalyzer` | Subfolder name for manual-analysis output |
| `normal` / `inverse` | `PerformanceAnalyzer` | Evaluation paths per mode |
| `apfd` | `APFDCalculator`, `main_calculate_apfd.py` | Input CSVs and fault percentile threshold |
| `moga` | `DiversityCalculator` | Initial population CSV for the GA |

### APFD fault threshold

```yaml
apfd:
  fault_percentile: 90   # tests above 90th-percentile difficulty count as faults
```

Increase this value to make the fault definition stricter (fewer faults, higher bar).

---

## Data Structure

Each use case expects this folder structure under `data/<uc>/`:

```
data/
└── uc1/
    ├── images/
    │   ├── train/   # Training images (.png)
    │   └── val/     # Validation images (.png)
    ├── labels/
    │   ├── train/   # YOLO label files (.txt)
    │   └── val/
    ├── masks/
    │   ├── train/   # Segmentation masks (.png)
    │   └── val/
    ├── merged_cycle0/           # Output of data merging (cycle 0)
    │   ├── images/
    │   ├── labels/
    │   ├── masks/
    │   └── cycle0_merged_stats.json
    └── normal_uc1_test_manual_analysis/
        ├── Elite_0/             # Selected hard cases per cycle
        └── Rejected_0/         # Discarded cases per cycle
```

---

## Pipeline Workflow

Run all scripts from the **project root directory**:

```bash
cd test-case-sifting
```

The full pipeline for one use case (e.g., `uc1`, cycle `0`) runs in this order:

```
Step 1  →  main_data_merging.py        Merge raw datasets
Step 2  →  main_clustering.py flatten  Extract & flatten scene features
Step 3  →  main_clustering.py cluster  K-means difficulty clustering
Step 4  →  main_ga.py                  MOGA test selection
Step 5  →  main_manual_analysis.py     Organize Elite / Rejected sets
Step 6  →  main_split_merge.py         Split for initial training
Step 7  →  main_train_perception.py    Train base YOLO model
Step 8  →  main_performance.py         Evaluate per difficulty bucket
Step 9  →  main_split_for_test.py      Split merged data for fine-tuning
Step 10 →  main_fine_tuning_perception.py  Fine-tune on hard cases
Step 11 →  main_gain_over_cycle.py     Compute cycle-over-cycle gains
Step 12 →  main_calculate_apfd.py      Compute APFD metric
```

---

## Script Reference

All scripts are in `scripts/` and must be run from the **project root**:

```bash
python scripts/<script_name>.py [arguments]
```

Valid values for `<use_case>`: `uc1` or `uc2`
Valid values for `<mode>`: `normal` or `inverse`
Valid values for `<cycle>`: integer (`0`, `1`, `2`, ...)

---

### `main_data_merging.py` — Merge raw datasets

Merges multiple source directories into a single dataset with globally unique IDs.

```bash
python scripts/main_data_merging.py <use_case> <cycle>
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |
| `cycle` | int | Target cycle number (e.g., `0`) |

```bash
# Example
python scripts/main_data_merging.py uc1 0
```

**Output:** `data/uc1/merged_cycle0/` with merged images, labels, masks and `cycle0_merged_stats.json`

---

### `main_clustering.py` — Feature extraction and clustering

Two modes: `flatten` extracts scene features; `cluster` runs K-means.

```bash
python scripts/main_clustering.py <action> <use_case> <cycle> [metric]
```

| Argument | Type | Description |
|----------|------|-------------|
| `action` | str | `flatten` or `cluster` |
| `use_case` | str | `uc1` or `uc2` |
| `cycle` | int | Cycle number |
| `metric` | str | *(optional)* Distance metric, e.g., `euclidean` |

```bash
# Step 1: flatten raw JSON into CSV with difficulty scores
python scripts/main_clustering.py flatten uc1 0

# Step 2: cluster by difficulty (K-means, k=3)
python scripts/main_clustering.py cluster uc1 0
```

**Output:**
- `data/uc1/uc1_cycle0_flattened_scene_features.csv`
- `data/uc1/uc1_cycle0_features_with_clusters_kmeans.csv`
- `outputs/plots/uc1_cycle0_kmeans_difficulty_segmentation.png`

---

### `main_ga.py` — MOGA test case selection (NSGA-II)

Runs the multi-objective genetic algorithm to select the most informative test cases (maximise difficulty + diversity).

```bash
python scripts/main_ga.py <use_case> <mode>
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |
| `mode` | str | `normal` (hard cases) or `inverse` (easy cases) |

```bash
python scripts/main_ga.py uc1 normal
```

**Output:**
- `uc1_normal_moga_selected_tests_run<n>.csv` — selected test IDs per run
- `uc1_normal_benchmark_results_final.csv` — aggregated results

---

### `main_manual_analysis.py` — Organize Elite / Rejected sets

Partitions test cases into **Elite** (selected) and **Rejected** buckets per difficulty level, inheriting previous-cycle Elites.

```bash
python scripts/main_manual_analysis.py <use_case> <mode> <cycle>
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |
| `mode` | str | `normal` or `inverse` |
| `cycle` | int | Current cycle (0, 1, 2, …) |

```bash
python scripts/main_manual_analysis.py uc1 normal 0
```

**Output:** `data/uc1/normal_uc1_test_manual_analysis/` with `Elite_0/`, `Rejected_0/` subfolders containing images, labels, and JSON feature files.

---

### `main_split_merge.py` — Split dataset for initial training

Splits a raw cycle dataset into train/val (80/20) and generates the YOLO `.yaml` config file.

```bash
python scripts/main_split_merge.py <use_case> <cycle>
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |
| `cycle` | int | Cycle number |

```bash
python scripts/main_split_merge.py uc1 0
```

**Output:** `data/uc1_dataset.yaml` (YOLO training config)

---

### `main_train_perception.py` — Train baseline YOLO model

Trains a YOLO11n-seg model from scratch on the cycle dataset.

```bash
python scripts/main_train_perception.py <use_case>
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |

```bash
python scripts/main_train_perception.py uc1
```

**Output:** `train_results/uc1/train_results_uc1/weights/best.pt`

> **Note:** Requires `yolo11n-seg.pt` at project root. Training uses 50 epochs, image size 640, batch 16.

---

### `main_performance.py` — Evaluate model per difficulty bucket

Evaluates the YOLO model separately on each difficulty bucket (Normal / Hard / Critical) and saves metrics.

```bash
python scripts/main_performance.py <use_case> <mode> [cycle]
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |
| `mode` | str | `normal` or `inverse` |
| `cycle` | int | *(optional)* Cycle number |

```bash
python scripts/main_performance.py uc1 normal 0
```

**Output:** `data/uc1/normal_uc1_test_manual_analysis/cycle0/cycle0_uc1_normal_evaluation_rq1_summary.csv`

---

### `main_split_for_test.py` — Split merged data for fine-tuning

Splits the merged cycle dataset into train/val for fine-tuning preparation.

```bash
python scripts/main_split_for_test.py <use_case> <cycle>
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |
| `cycle` | int | Cycle number |

```bash
python scripts/main_split_for_test.py uc1 0
```

---

### `main_fine_tuning_perception.py` — Fine-tune YOLO on hard cases

Fine-tunes the YOLO model on the Rejected (hard) cases from the previous cycle.

```bash
python scripts/main_fine_tuning_perception.py <use_case> <cycle>
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |
| `cycle` | int | Current cycle (≥ 1) — uses `Rejected_{cycle-1}` |

```bash
python scripts/main_fine_tuning_perception.py uc1 1
```

**Output:** `fine-tuning/uc1/cycle1_uc1/weights/best.pt`

> Learning rate decays with cycle: `lr = 0.01 / 10^cycle`

---

### `main_gain_over_cycle.py` — Compute cycle-over-cycle performance gains

Computes absolute and relative gains in mAP50-95 and F1 between consecutive cycles, and the reduction factor in critical failures.

```bash
python scripts/main_gain_over_cycle.py <use_case>
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |

```bash
python scripts/main_gain_over_cycle.py uc1
```

**Output:** JSON gain reports + `outputs/plots/uc1_apfd_curve_cycle*.png`, `uc1_violin_plot_difficulty.png`

---

### `main_gain_over_cycle_withplot.py` — Gains with combined plot

Extended version of the above with an annotated combo chart (mAP + F1 + critical failures bar).

```bash
python scripts/main_gain_over_cycle_withplot.py <use_case> [cycle_start] [cycle_end]
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |
| `cycle_start` | int | *(optional, default 0)* |
| `cycle_end` | int | *(optional, default 2)* |

```bash
python scripts/main_gain_over_cycle_withplot.py uc1 0 2
```

**Output:** `outputs/plots/uc1_performance_combo_plot.png`

---

### `main_calculate_apfd.py` — Compute APFD metric

Calculates the **Average Percentage of Faults Detected** for naive and elite test orderings.

```bash
python scripts/main_calculate_apfd.py <use_case>
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |

```bash
python scripts/main_calculate_apfd.py uc1
```

**Output:** `uc1_apfd_analysis_summary.csv`

---

### `main_vision_complexity.py` — Vision-based complexity scoring

Computes the vision complexity fitness score for test cases from a JSON file.

```bash
python scripts/main_vision_complexity.py
```

> Edit `file_path` inside the script to point to your JSON data file before running.

---

### `diversity_performance.py` — Statistical diversity analysis

Computes and compares diversity metrics (Cliff's Delta effect size) between random and optimised test selections.

```bash
python scripts/diversity_performance.py <use_case> <cycle>
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |
| `cycle` | int | Cycle number |

```bash
python scripts/diversity_performance.py uc1 0
```

---

### `naive_cluster_performance.py` — Cluster-based naive evaluation

Evaluates model performance per cluster (Normal/Hard/Critical) using a naive YOLO inference on sorted subsets.

```bash
python scripts/naive_cluster_performance.py <use_case> <cycle>
```

| Argument | Type | Description |
|----------|------|-------------|
| `use_case` | str | `uc1` or `uc2` |
| `cycle` | int | Cycle number |

```bash
python scripts/naive_cluster_performance.py uc1 0
```

**Output:** `data/uc1_cycle0_cluster_performance_results.csv`

---

### `model_test.py` — Quick model inference test

Runs YOLO inference on a single image and prints perception difficulty. Edit the model and image paths inside the script.

```bash
python scripts/model_test.py
```

---

### `test-prioritization.py` — Diversity-driven prioritization

Ranks test cases by cosine diversity using a JSON data file.

```bash
python scripts/test-prioritization.py <n_tests>
```

| Argument | Type | Description |
|----------|------|-------------|
| `n_tests` | int | Number of test cases to select |

```bash
python scripts/test-prioritization.py 100
```

---

## Outputs

| Location | Content |
|----------|---------|
| `outputs/plots/` | All generated figures (PNG): clustering segmentations, APFD curves, violin plots, elbow plots |
| `data/<uc>/` | CSVs with difficulty scores, cluster assignments, evaluation summaries |
| `train_results/<uc>/` | YOLO training logs (`results.csv`) and tensorboard data |
| `fine-tuning/<uc>/` | Fine-tuned model weights per cycle |
| `models/` | Saved YOLO `.pt` model files |

---

## Reproducibility Notes

### Portability of `config.yaml`

All paths in `config.yaml` are **relative to the project root** and require no modification after cloning. The only thing you must do is place your raw simulation data in the expected location:

```
test-case-sifting/
└── data/
    └── raw/
        ├── uc1/
        │   ├── dir_1/
        │   │   ├── dir_1_stats.json
        │   │   ├── images/
        │   │   └── labels/
        │   └── dir_2/
        │       └── ...
        └── uc2/
            └── ...
```

If your raw data lives elsewhere, update only the two `base_path` values in `config.yaml`:

```yaml
merging:
  uc1:
    base_path: "path/to/your/raw/uc1/data"
  uc2:
    base_path: "path/to/your/raw/uc2/data"
```

All other paths are generated automatically by the pipeline.

### Random seeds

The GA optimiser uses a fixed seed for reproducibility:

```python
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
```

This is set inside `scripts/main_ga.py` and `scripts/main_ga copy.py`.

### Python version

Tested with **Python 3.10**. YOLO and PyTorch versions are pinned in `requirements.txt`. For GPU training, ensure your CUDA version is compatible with `torch==2.9.0`.

### Minimal hardware

| Task | Minimum |
|------|---------|
| Feature extraction / clustering | 8 GB RAM, CPU |
| YOLO training (50 epochs, 640px) | 8 GB VRAM (GPU) |
| MOGA (100 pop, 200 gen) | 16 GB RAM, CPU |
| Fine-tuning | 8 GB VRAM (GPU) |

### Running the full pipeline (one use case, cycle 0 → 1)

```bash
USE_CASE=uc1

# Cycle 0: initial training
python scripts/main_data_merging.py $USE_CASE 0
python scripts/main_clustering.py flatten $USE_CASE 0
python scripts/main_clustering.py cluster $USE_CASE 0
python scripts/main_ga.py $USE_CASE normal
python scripts/main_manual_analysis.py $USE_CASE normal 0
python scripts/main_split_merge.py $USE_CASE 0
python scripts/main_train_perception.py $USE_CASE
python scripts/main_performance.py $USE_CASE normal 0

# Cycle 1: fine-tuning on hard cases
python scripts/main_split_for_test.py $USE_CASE 0
python scripts/main_fine_tuning_perception.py $USE_CASE 1
python scripts/main_performance.py $USE_CASE normal 1
python scripts/main_gain_over_cycle.py $USE_CASE
python scripts/main_calculate_apfd.py $USE_CASE
```

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{testcasesifting2025,
  author = {André Nguimbous},
  title  = {Risk-Aware, Novelty-First Test Sifting for Continuous Evaluation
of Manipulator Vision Models},
  year   = {2025},
  url    = {https://github.com/andre07cloud/test-case-sifting}
}
```
