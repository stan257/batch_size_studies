# Batch Size Studies

This project provides a robust framework for conducting and analyzing machine learning experiments, with a focus on studying the effects of hyperparameters like batch size, learning rate, and model parameterization.

## Key Features

- **Centralized Configuration**: All experiment parameters and hyperparameter grids are defined in a single source of truth, `configs.py`.
- **Automated Sweeps**: Simple command-line scripts allow for running large, multi-CPU hyperparameter sweeps.
- **Resumable Training**: A robust checkpointing system automatically saves progress, allowing long-running experiments to be stopped and resumed without losing work.
- **Powerful Analysis Tools**: A suite of data utilities in `src/batch_size_studies/data_utils.py` for filtering, smoothing, and processing raw results for analysis.
- **Automated Reporting**: Generate self-contained HTML reports with publication-ready visualizations (heatmaps, loss curves) with a single command.

## Setup

1.  **Clone the repository and navigate into it:**
    ```bash
    git clone <repository-url>
    cd batch_size_studies
    ```

2.  **Create and activate a Python environment:**
    A virtual environment (like conda or venv) is recommended.
    ```bash
    conda create -n batch_size_env python=3.11
    conda activate batch_size_env
    ```

3.  **Install dependencies:**
    Install the project in editable mode. This will also install all required packages from `pyproject.toml`.
    ```bash
    pip install -e .
    ```

4.  **(Optional) Prepare the MNIST-1M Dataset:**
    To run experiments on the `mnist1m` dataset, download the 10 raw `.zip` files from the source and place them in `data/mnist1m/raw/`. Then, process them by running:
    ```bash
    python scripts/process_mnist1m.py
    ```
    This creates the `data/mnist1m/mnist1m.npz` file required for training.

## Workflow

The typical workflow involves defining, running, and analyzing experiments.

### 1. Define an Experiment
All experiment configurations are defined as dataclasses in `configs.py`. This is the single source of truth for what to run.

### 2. Run Experiments

Use the scripts in the `scripts/` directory to execute experiment sweeps. Results and checkpoints are saved to the `experiments/` directory. If a run is interrupted, it will automatically resume from the last checkpoint when you run the script again.

#### Main Experiments

Use `scripts/run_experiments.py` to run the main suite of experiments.

*   **Run all experiments:**
    ```bash
    python scripts/run_experiments.py
    ```

*   **Run one or more specific experiments by name:**
    ```bash
    python scripts/run_experiments.py -n <experiment_name_1> <experiment_name_2>
    ```
    *(See `get_main_experiment_configs()` in `configs.py` for available names.)*

*   **Override parameters for a quick test:**
    ```bash
    # Run with 5 epochs instead of the configured value
    python scripts/run_experiments.py -n <experiment_name> -o num_epochs=5
    ```
*   **Run without saving results (for debugging):**
    ```bash
    python scripts/run_experiments.py --no-save
    ```


#### Small µP Experiments

A smaller suite of experiments can be run in parallel.

*   **Run all small µP experiments (using all available CPU cores):**
    ```bash
    python scripts/run_small_muP_experiments.py
    ```

*   **Limit the number of parallel workers:**
    ```bash
    python scripts/run_small_muP_experiments.py --max_workers 4
    ```

#### Experimental Sweeps

The `scripts/run_experimental_sweep.py` script is designed for targeted sweeps over `gamma` and `eta` for a **fixed batch size**, which is useful for sanity checks and detailed analysis of specific regions of the hyperparameter space.

*   **Run a sweep for a fixed batch size:**
    ```bash
    # Example: Sweep gamma and eta for a fixed batch size of 256
    python scripts/run_experimental_sweep.py --batch-size 256
    ```

*   **Customize sweep ranges and resolution:**
    ```bash
    python scripts/run_experimental_sweep.py --batch-size 256 --gamma-range 3 --eta-range 10
    ```

### 3. Generate Reports and Analyze Results

After experiments have run, use the analysis scripts to process the raw data and generate visualizations.

#### Main Experiment Reports

Generate a self-contained HTML report with visualizations from the main experiments.

*   **Generate a report for all experiments:**
    ```bash
    python scripts/generate_reports.py
    ```

*   **Generate a report for specific experiments:**
    ```bash
    python scripts/generate_reports.py -n <experiment_name_1> <experiment_name_2>
    ```

*   **Customize included plots:**
    ```bash
    python scripts/generate_reports.py --plots heatmap_batch losscurve_temp_samples
    ```
    *(See `PLOT_REGISTRY` in `scripts/generate_reports.py` for available plots.)*

Reports are saved in the `reports/` directory.

#### Small µP Experiment Analysis

Analysis for the small µP experiments is a two-step process using `scripts/analyze_small_muP_results.py`.

1.  **Aggregate Results**: Process the raw experiment outputs into a single analysis file.
    ```bash
    python scripts/analyze_small_muP_results.py --mode analyze
    ```

2.  **Generate Plots**: Create specific loss curve plots from the aggregated data.
    ```bash
    # Example: Plot loss curves for a specific configuration.
    # Note: Requires experiment_type, gamma, batch_size, and eta.
    python scripts/analyze_small_muP_results.py --mode plot \
        --experiment_type "fixed_time_poly_teacher" \
        --gamma 1.0 \
        --batch_size 64 \
        --eta 0.1
    ```
    Plots are saved as `.png` images in the `plots/` directory.

## Codebase Tour

Key files and directories:

*   `configs.py`: **The single source of truth.** Defines all experiment configurations and hyperparameter grids.
*   `scripts/`: Contains all runnable scripts for experiments, analysis, and data processing.
*   `src/batch_size_studies/`: The core library code.
    *   `mnist_training.py` & `synthetic_training.py`: Contain the core JAX/Optax training loops for the different experiment types.
    *   `data_utils.py`: A collection of functions for post-processing experiment data.
*   `experiments/`: Default output directory for all raw experiment results (`.pkl` files) and checkpoints.
*   `reports/`: Default output directory for generated HTML reports.
*   `plots/`: Default output directory for individual plots.
*   `tests/`: Contains unit and integration tests for the framework.