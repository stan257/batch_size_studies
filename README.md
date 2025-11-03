# Batch Size Studies

Framework for exploring how `batch_size`, `learning_rate`, and model scaling interact in SP and µP regimes. The tooling runs reproducible sweeps, tracks checkpoints, and keeps results easy to analyze so you can focus on the science.

## Quick start for sweeps

1. **Install once** (any Python ≥3.10):
   ```bash
   pip install -e .
   ```

2. **Datasets**
   - MNIST downloads automatically via `tensorflow_datasets`.
   - For MNIST-1M, generate `data/mnist1m/mnist1m.npz` with `python scripts/process_mnist1m.py` after sourcing the raw files.

3. **Launch a sweep**
   ```bash
   python scripts/run_experiments.py --name mnist1m_mup_SGD_gamma1p0 ...
   ```
   Useful flags: `--no-save` (dry run), `--max-eval-samples 2000` (faster validation), `--eta-stability-depth 3` (stop once three stable etas appear). Results are stored in `experiments/<experiment_type>/` with losses, metadata, and checkpoints.

4. **Inspect results**
   - Use notebooks in `notebooks/` or the utilities in `scripts/analyze_small_muP_results.py` and `scripts/generate_reports.py` for plots and HTML summaries.

## Other sweeps at a glance

| Script | What it explores |
| --- | --- |
| `scripts/run_width_sweep.py` | Width transfer experiments at fixed `(batch_size, eta)`. |
| `scripts/run_small_muP_experiments.py` | Finds minimum widths that exhibit µP behaviour. |

All scripts ultimately call `run_experiment_sweep`, so any experiment defined in `configs.py` shows up everywhere automatically.

## Repository map

```
src/batch_size_studies/
├── configs.py         # canonical experiment definitions and grids
├── experiments.py     # dataclasses for synthetic & MNIST families
├── runner.py          # orchestrates sweeps, checkpointing, resumption
├── trainer.py         # TrialRunner hierarchy (MNIST vs synthetic)
├── models.py          # SP / µP models
├── data_loading.py    # dataset access (MNIST, MNIST-1M)
└── plotting_utils.py  # loss heatmaps, stability curves
scripts/               # command-line entry points for sweeps & reports
tests/                 # unit + integration suites (reproducibility, CLI)
```

## Adding a new study

1. Create the dataclass in `experiments.py` (inherit the SP/µP mixins).
2. Register it in `configs.py` with its hyperparameter grid.
3. Run `python scripts/run_experiments.py --name <your_experiment>`.

The runner handles checkpoints, resumability, and logging; plots and reports work out of the box once results exist.

## Getting help

Open a GitHub issue or discussion if you need to hook in a new dataset or want to contribute additional analysis scripts—contributions that sharpen research workflows are welcome.
