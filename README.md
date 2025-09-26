# Batch Size Studies

This repository contains a framework for systematically studying the effects of hyperparameters—primarily `batch_size` and `learning_rate`—on the training dynamics of neural networks, with a focus on Standard (SP) and µ-Parametrization (µP).

The codebase is designed to be modular and extensible, allowing for easy definition of new experiments, models, and analysis routines.

## Installation

1.  **Clone the repository and navigate into it:**
    ```bash
    git clone https://github.com/stan257/batch_size_studies.git
    cd batch_size_studies
    ```

2.  **Create and activate a Python environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    Install the project in editable mode, which is recommended for development:
    ```bash
    pip install -e .
    ```

## Data Setup

This project uses the MNIST and MNIST-1M datasets.

*   The standard **MNIST** dataset will be downloaded automatically by `tensorflow_datasets` on the first run.
*   The **MNIST-1M** dataset must be processed first. After downloading the raw data, run the provided script to create a consolidated `.npz` file:
    ```bash
    # (Instructions on where to get the raw MNIST-1M data)
    python scripts/process_mnist1m.py
    ```

## Usage

The `scripts/` directory contains high-level scripts to run experiments and generate reports.

#### Running Experiments

The main script for running hyperparameter sweeps is `run_experiments.py`. You can run all defined experiments or select specific ones by name.

To run all main experiments:
    ```bash
    python scripts/run_experiments.py
    ```

To run a specific experiment (e.g., `mnist_classification_mup`):
    ```bash
    python scripts/run_experiments.py --name mnist_classification_mup
    ```

Other specialized runner scripts are also available:
*   `run_small_muP_experiments.py`: Runs experiments designed to find the smallest widths at which µP properties emerge.
*   `run_width_sweep.py`: Sweeps over model width for a fixed `batch_size` and `eta`.
*   `run_experimental_sweep.py`: Runs a 2D sweep over `gamma` and `eta` for a fixed batch size.

For example, to run a width sweep:
```bash
python scripts/run_width_sweep.py \
    --base_experiment mnist1m_sampled_mup_L3_N64_gamma1p0 \
    --widths 64 128 256 512 \
    --batch_size 256 \
    --eta 0.01
```

#### Analyzing Results and Generating Reports

After running experiments, you can analyze the results and generate plots or HTML reports.

1.  **Analyze Raw Results**:
    This script processes the raw experiment data into a structured format for easier plotting.
    ```bash
    python scripts/analyze_small_muP_results.py --mode analyze
    ```

2.  **Generate Plots**:
    You can generate individual plots from the analyzed data.
    ```bash
    python scripts/analyze_small_muP_results.py \
        --mode plot \
        --experiment_type mnist1m_sampled_classification \
        --gamma 1.0 \
        --batch_size 256 \
        --eta 0.01
    ```

3.  **Generate HTML Report**:
    This creates a self-contained HTML report with embedded plots for easy sharing.
    ```bash
    python scripts/generate_reports.py
    ```

## Project Structure

The codebase is organized into three main components:

*   `src/batch_size_studies/`: The core library.
    *   `configs.py`: Defines all experiment configurations and hyperparameter grids. This is the main entry point for defining new studies.
    *   `experiments.py`: Contains the `dataclass` definitions for different experiment types.
    *   `runner.py`: A unified, high-level function (`run_experiment_sweep`) that orchestrates all experiment runs.
    *   `trainer.py`: Contains the `TrialRunner` class hierarchy, which encapsulates the detailed training and evaluation logic for each experiment family (e.g., `MNISTTrialRunner`, `SyntheticTrialRunner`).
    *   `models.py`: Defines the MLP model and its parameterizations (SP and µP).
    *   `data_loading.py`: Handles loading and pre-processing of datasets.
    *   `checkpoint_utils.py`: Manages saving and loading for resumability and analysis.

*   `scripts/`: Executable scripts that provide the command-line interface for running experiments and analysis. These scripts use the core library.

*   `tests/`: The `pytest` test suite, including unit tests, integration tests, and reproducibility tests to ensure the correctness of the training logic.
