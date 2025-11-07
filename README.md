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
├── configs.py          # grids used in the main sweeps
└── plotting_utils.py   # plotting helpers used in notebooks
scripts/                # CLI entry points for sweeps & post-processing
notebooks/              # analysis notebooks accompanying the paper
```

## Running the sweeps

1. Install the package (Python ≥3.10):
   ```bash
   pip install -e .
   ```

2. Prepare datasets:
   * MNIST downloads automatically through `tensorflow_datasets`.
   * MNIST‑1M requires `python scripts/process_mnist1m.py` to convert the diffusion-generated set into `data/mnist1m/mnist1m.npz`.

3. Launch a sweep defined in `configs.py`:
   ```bash
   python scripts/run_experiments.py --name mnist1m_mup_SGD_gamma1p0
   ```
   Optional flags:
   * `--no-save` to dry-run in place (useful while inspecting behaviour).
   * `--max-eval-samples N` to limit validation cost.
   * `--eta-stability-depth K` to stop exploring learning rates once K consecutive stable learning rates are observed per batch size.
   * `--num-processes M` to run up to M sweeps in parallel (useful on clusters; defaults to sequential execution).

4. Results land in `experiments/<experiment_type>/`:
   * `results_*.pkl` store loss histories and failure logs.
   * `_weights.pkl` capture initial parameters, deltas, and sweep metadata.
   * `_checkpoints/` contain live resume files, cleaned automatically when runs finish successfully.

5. Use the notebooks in `notebooks/` or scripts in `scripts/` to regenerate figures and summaries.

The test suite (`pytest`) focuses on regression coverage for checkpoints, runner orchestration, and iterator behaviour to ensure the code reproduces the results described in the accompanying manuscript.
