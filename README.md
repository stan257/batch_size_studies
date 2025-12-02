# Batch Size Studies

This repository captures the code used to generate our batch size and learning rate sweeps across synthetic teachers, MNIST, and MNIST‑1M. The emphasis is on reproducible sweeps rather than a polished library; the structure below is meant to help readers replicate our experiments or inspect intermediate artefacts.

## Repository layout

```
src/batch_size_studies/
├── constants.py        # shared configuration, seeds, and eval defaults
├── experiments.py      # experiment configurations and dataset preparation
├── runner.py           # sweep orchestration, resumability, stability checks
├── trainer.py          # TrialRunner implementations (MNIST vs. synthetic)
├── data_iterators.py   # offline vs. online data presentation paradigms
├── models.py           # SP/µP MLPs, linear baselines, centered wrapper
├── checkpoint_utils.py # checkpoints, metadata, legacy compatibility
├── storage_utils.py    # atomic pickle helpers
├── configs.py          # surfaces registered experiments + grids
├── registered_experiments.py  # declarative sweep specs
├── definitions.py      # enums and run key types shared across modules
├── protocols.py        # shared interfaces (TrialRunner, DataIterator, ModelProtocol)
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

3. List available sweeps:
   To see all experiments registered in the framework, use the `list` command.
   ```bash
   python scripts/run_experiments.py list
   ```
   You can also filter this list, e.g., `python scripts/run_experiments.py list --optimizer Adam`. Add `--list-overrides` to print the supported `--override key=value` options.

4. Launch a sweep:
   Use the `run` command to execute an experiment from the catalog.
   ```bash
   python scripts/run_experiments.py run --name mnist1m_mup_SGD_gamma1p0
   ```
   Optional flags for the `run` command:
   * `--no-save` to dry-run in place (useful while inspecting behaviour).
   * `--override <KEY>=<VALUE>` to dynamically change a parameter (e.g., `num_epochs=10`). Run `python scripts/run_experiments.py list --list-overrides` to see the available keys.
   * `--max-eval-samples N` to limit validation cost.
   * `--eta-stability-depth K` to stop exploring learning rates once K consecutive stable learning rates are observed per batch size.
   * `--num-processes M` to run up to M sweeps in parallel (useful on clusters; defaults to sequential execution).
   * `--save-interstitial-snapshots` / `--no-save-interstitial-snapshots` to toggle dense weight snapshots between checkpoints.
   * `--save-epoch-snapshots` / `--no-save-epoch-snapshots` to control whether fixed-data synthetic runs store weights at every epoch boundary.
   * `--dry-run` (with optional `--dry-run-steps`) to execute a single `(B, η)` for a few steps without saving results—ideal for debugging shapes/log output before launching long sweeps.

5. Results land in `experiments/<experiment_type>/`:
   * `results_*.pkl` store loss histories and failure logs.
   * `_weights.pkl` capture initial parameters, deltas, and sweep metadata.
   * `_checkpoints/` contain live resume files, cleaned automatically when runs finish successfully.

6. Use the notebooks in `notebooks/` (together with `src/batch_size_studies/plotting_utils.py`) to regenerate figures and bespoke summaries for each experiment family.

The test suite (`pytest`) focuses on regression coverage for checkpoints, runner orchestration, and iterator behaviour to ensure the code reproduces the results described in the accompanying manuscript.

## Pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to run Ruff (lint + format), basic hygiene checks, and a quick smoke test suite before every commit.

To enable the hooks locally:

```bash
pip install pre-commit
pre-commit install
```

The smoke suite runs `pytest -m smoke`, which exercises a tiny synthetic sweep and key CLI commands, so it stays fast while still catching obvious regressions.

This project is licensed under the MIT License - see the LICENSE file for details.
