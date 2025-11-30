# Code Summary

This repository captures the code we use to run batch-size/learning-rate sweeps across synthetic and MNIST-style workloads. The notes below are meant to help contributors find the relevant module quickly.

---

## Experiments (`experiments.py`)
* Dataclasses that hold experiment configuration (model, dataset construction, sampling regime, etc.).
* Handle dataset prep (MNIST downloads, MNIST‑1M subsampling, synthetic teacher generation) and record sweep metadata for later analysis.
* Provide hooks such as `should_skip_batch_size`, `compute_num_steps`, and `is_classification()` so the runner and plotting code behave correctly.

## Definitions & Paths (`definitions.py`, `paths.py`)
* Shared enums (`LossType`, `OptimizerType`, `Parameterization`) and the `RunKey` dataclass live in `definitions.py`.
* `paths.py` resolves `PROJECT_ROOT`, `EXPERIMENTS_DIR`, and `DATA_DIR`, creating those directories so sweep artefacts always land in predictable locations.

## Experiment Registry (`experiment_registry.py`, `registered_experiments.py`, `configs.py`)
* Experiment specs are declared once (optimizer, loss, experiment family, kwargs) and registered via builders.
* `experiment_registry.py` stores the list and enforces unique names; `configs.py` materializes them and exposes the canonical `(batch_size, η)` grids.
* Adding a study = implement the dataclass + add a builder; CLI and notebooks will pick it up automatically.

## Runner (`runner.py`)
* Drives the `(B, η)` grid: loads prior results, checks if a run is complete, and dispatches work through `TrialRunner`.
* Persists losses/failures/metadata so long sweeps can resume cleanly.
* Implements ETA-stability early stopping and provides the CLI entry point filters (by experiment family, optimizer, loss).

## Trainer (`trainer.py`)
* Shared protocol for a single run: resume (if checkpoint exists), create data iterator, train, log metrics, snapshot weights.
* `EpochBasedTrialRunner` centralizes fixed-dataset bookkeeping (steps per epoch, iterator wiring, epoch hooks) so MNIST and other offline experiments only define their per-epoch logic.
* MNIST and synthetic subclasses still define task-specific losses/metrics; everything uses the `½ ∥·∥²` MSE normalization.
* Caches JIT-compiled update/eval functions per model instance to avoid recompilation across sweeps.
* `loss_history` stores post-update minibatch losses only (no step-0 entry), so trajectories line up across runs with the same init/data order.

## Training Utilities (`training_utils.py`)
* Builds the base Optax transforms (SGD identity or Adam) without applying η so the trainer can apply sweep-specific learning rates after µP scaling.
* Provides helpers that reverse µP/γ scaling (and optional legacy factors) when comparing empirical η grids to theory.

## Models (`models.py`)
* `MLP` (SP vs. μP parameterizations) and `LinearModel` definitions used by experiments.
* `CenteredModel` wrapper subtracts the network output at initialization so loss/metrics track feature learning rather than raw logits.
* All models expose `init_params` + `__call__`, matching the simple protocol expected by the trainer.

## Data Presentation (`data_iterators.py`)
* Epoch-based iterator: reproducible shuffling, resume mid-epoch, handles dict datasets (MNIST) or tuple datasets (synthetic fixed-data).
* Online iterator: regenerates synthetic batches on demand, keeping track of the PRNG seed so checkpoints resume the same data stream.

## Data Loading (`data_loading.py`)
* Thin wrappers for tfds MNIST download and local MNIST‑1M `.npz` loading (with scaling to `[0, 1]` and channel-axis handling).
* Keeps dataset preparation logic out of notebooks/scripts; experiments can inject alternate loaders via kwargs when needed.

## Data Utilities (`data_utils.py`)
* Helper functions to pull loss histories out of stored result dicts, smooth/transform them, and filter by `(B, η, η/B)` before plotting.

## Plotting (`plotting_utils.py`)
* Heatmaps over `(B, η)` using extracted final metrics plus grouped loss-curve plots with options for effective steps or samples-seen axes.
* Supports overlaying divergence/theory curves and critical batch-size markers for notebook analysis.

## Checkpointing & Storage (`checkpoint_utils.py`, `storage_utils.py`)
* Atomic live-checkpoint saves + weight-snapshot deltas (with file locks) so interrupted sweeps resume without corrupting state.
* Store initial params and sweep metadata alongside weight deltas for downstream notebooks.

## Hessian Tooling (`hessian.py`, `hessian_evaluator.py`)
* `hessian.py` implements HVP-based power iteration, Hutchinson trace estimates, and spectral density routines.
* `hessian_evaluator.py` reloads saved weights + metadata, rebuilds the exact iterator order, and feeds batches into the Hessian routines for any stored run.

## Spectral Data CLI (`scripts/gather_hessian_spectra.py`)
* Command-line tool that lists available snapshot steps and runs `HessianEvaluator` on selected `(B, η, step)` combinations.
* Stores eigenvalues/trace in `spectral_data/<experiment_type>/<experiment>_spectra.pkl`, mirroring the experiment serialization layout.

## Integration & Tests
* Coverage for checkpoints/metadata, iterator behavior, sweep orchestration, Hessian evaluation, and linear/noisy teacher configs.
* Registry tests ensure spec builders stay valid and names remain unique.

---

In practice: define the experiment dataclass, register it, and launch `run_experiment_sweep`. Data loading, checkpoints, stability tracking, and plotting work out of the box across studies.
