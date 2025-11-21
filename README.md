# Batch Size Studies

This repository captures the code used to generate our batch size and learning rate sweeps across synthetic teachers, MNIST, and MNIST‑1M. The emphasis is on reproducible sweeps rather than a polished library; the structure below is meant to help readers replicate our experiments or inspect intermediate artefacts.

## Repository layout

```
src/batch_size_studies/
├── experiments.py      # experiment configurations and dataset preparation
├── runner.py           # sweep orchestration, resumability, stability checks
├── trainer.py          # TrialRunner implementations (MNIST vs. synthetic)
├── data_iterators.py   # offline vs. online data presentation paradigms
├── models.py           # SP/µP MLPs, linear baselines, centered wrapper
├── checkpoint_utils.py # checkpoints, metadata, legacy compatibility
├── storage_utils.py    # atomic pickle helpers
├── configs.py          # surfaces registered experiments + grids
├── registered_experiments.py  # declarative sweep specs
└── plotting_utils.py   # plotting helpers used in notebooks
scripts/                # CLI entry points for sweeps & post-processing
notebooks/              # analysis notebooks accompanying the paper
```

## Running the sweeps

1. Install the package (Python ≥3.10):
   ```bash
   pip install -e .
   ```
   For a reproducible setup matching the tested environment (Python 3.12, JAX 0.7.2), create a conda env from `environment.yml`:
   ```bash
   conda env create -f environment.yml
   conda activate batch-size-studies
   pip install -e .
   ```
   *If you target a non-CPU accelerator, adjust the JAX/TensorFlow wheels in `environment.yml` (e.g., CUDA/ROCm builds) before installing.*

2. Prepare datasets:
   * MNIST downloads automatically through `tensorflow_datasets`.
   * MNIST‑1M requires `python scripts/process_mnist1m.py` to convert the diffusion-generated set into `data/mnist1m/mnist1m.npz`.

3. Launch a sweep from the registered catalog:
   ```bash
   python scripts/run_experiments.py --name mnist1m_mup_SGD_gamma1p0
   ```
   Optional flags:
   * `--no-save` to dry-run in place (useful while inspecting behaviour).
   * `--max-eval-samples N` to limit validation cost.
   * `--eta-stability-depth K` to stop exploring learning rates once K consecutive stable learning rates are observed per batch size.
   * `--num-processes M` to run up to M sweeps in parallel (useful on clusters; defaults to sequential execution).
   * `--save-interstitial-snapshots` / `--no-save-interstitial-snapshots` to toggle dense weight snapshots between checkpoints (useful when trading analysis granularity for speed).
   * `--save-epoch-snapshots` / `--no-save-epoch-snapshots` to control whether fixed-data synthetic runs store weights at every epoch boundary.

4. Results land in `experiments/<experiment_type>/`:
   * `results_*.pkl` store loss histories and failure logs.
   * `_weights.pkl` capture initial parameters, deltas, and sweep metadata.
   * `_checkpoints/` contain live resume files, cleaned automatically when runs finish successfully.

5. Use the notebooks in `notebooks/` (together with `src/batch_size_studies/plotting_utils.py`) to regenerate figures and bespoke summaries for each experiment family.

The test suite (`pytest`) focuses on regression coverage for checkpoints, runner orchestration, and iterator behaviour to ensure the code reproduces the results described in the accompanying manuscript.
