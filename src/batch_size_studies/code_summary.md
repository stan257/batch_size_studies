# Code Summary

This repository captures the code we use to run batch-size/learning-rate sweeps across synthetic and MNIST-style workloads. The notes below are meant to help contributors find the relevant module quickly.

---

## Research Loop View
* **Design a study** in `studies/` and `experiment_types/` (what tasks/hyperparameters to run).
* **Run sweeps** through `scripts/run_experiments.py`, which delegates to `cli.py` and `engine/runner.py`.
* **Resume and preserve state** through checkpoints and weight snapshots in `engine/checkpoint_utils.py`.
* **Analyze outcomes** with `plotting_utils.py`, notebooks, and optional Hessian tooling under `spectral/`.

## Experiment Types (`experiment_types/`)
* `experiment_types/base.py` defines shared mixins (`MLPStudentExperiment`, `LinearStudentExperiment`) and `ExperimentBase` with filename IO helpers.
* `experiment_types/synthetic.py` contains all synthetic families (fixed-time, fixed-data, linear/noisy teachers, MLP teachers).
* `experiment_types/mnist.py` houses MNIST and MNIST‑1M variants plus dataset helpers.
* `experiments.py` remains as a thin compatibility shim that re-exports these classes for legacy imports/pickles.

## Constants (`constants.py`)
* Consolidated home for “magic numbers” that define evaluation seed offsets, batch sizes, and shared defaults consumed by trainers/experiments/runners.
* Keeps PRNG streams disjoint and makes it easy to tweak evaluation behaviour without hunting through multiple modules.

## Protocols (`protocols.py`)
* Defines the shared interfaces used by the sweep engine (`TrialRunner`, `DataIterator`, `ModelProtocol`).
* These interfaces keep experiments, training loops, and post-hoc analysis code aligned.

## Engine (`engine/`)
* `engine/runner.py` orchestrates complete sweeps over `(B, η)` and handles resume/skip logic.
* `engine/trainer.py` executes one trial at a time and records task-specific metrics.
* `engine/checkpoint_utils.py`, `engine/storage_utils.py`, and `engine/data_iterators.py` preserve exact run state so interrupted experiments remain reproducible.

## Runner (`engine/runner.py`)
* Drives the `(B, η)` grid: loads prior results, checks if a run is complete, and dispatches work through `TrialRunner`.
* Persists losses/failures/metadata (including the git commit) so long sweeps can resume cleanly.
* Implements ETA-stability early stopping and a CLI `--dry-run` mode for quick debugging of a single `(B, η)` combo.

## Main Experiment Runner (`scripts/run_experiments.py`)
* The primary command-line tool for managing experiment sweeps.
* Uses a subcommand structure: `list` to discover available experiments and `run` to execute them.
* Supports filtering by name, type, optimizer, and loss, as well as parallel execution and dynamic parameter overrides.

## Trainer (`engine/trainer.py`)
* Provides concrete implementations of `TrialRunner` (e.g., `EpochBasedTrialRunner`, `MNISTTrialRunner`).
* Handles the shared protocol for a single run: resume (if checkpoint exists), create data iterator, train, log metrics, snapshot weights.
* `EpochBasedTrialRunner` centralizes fixed-dataset bookkeeping (steps per epoch, iterator wiring, epoch hooks) so MNIST and other offline experiments only define their per-epoch logic.
* MNIST and synthetic subclasses still define task-specific losses/metrics; everything uses the `½ ∥·∥²` MSE normalization.
* Caches JIT-compiled update/eval functions per model instance to avoid recompilation across sweeps.
* `loss_history` stores post-update minibatch losses only (no step-0 entry), so trajectories line up across runs with the same init/data order.

## Training Utilities (`engine/training_utils.py`)
* Builds the base Optax transforms (SGD identity or Adam) without applying η so the trainer can apply sweep-specific learning rates after µP scaling.
* Provides helpers that reverse µP/γ scaling (and optional legacy factors) when comparing empirical η grids to theory.

## Models (`models.py`)
* `MLP` (SP vs. μP parameterizations) and `LinearModel` definitions used by experiments.
* `CenteredModel` wrapper subtracts the network output at initialization so loss/metrics track feature learning rather than raw logits.
* All models expose `init_params` + `__call__`, matching the simple protocol expected by the trainer.

## Data Presentation (`engine/data_iterators.py`)
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

## Checkpointing & Storage (`engine/checkpoint_utils.py`, `engine/storage_utils.py`)
* Atomic live-checkpoint saves + weight-snapshot deltas (with file locks) so interrupted sweeps resume without corrupting state.
* Store initial params and sweep metadata alongside weight deltas for downstream notebooks.

## Study Catalogs (`studies/`)
* Study definitions are split by domain (`studies/linear_teacher_catalog.py`, `studies/mnist1m_catalog.py`).
* `studies/catalog.py` aggregates them for registry side effects used by `configs.py`.
* In practice, this is where you add new paper-specific sweep families.

## Hessian Tooling (`spectral/hessian.py`, `spectral/hessian_evaluator.py`)
* `hessian.py` implements HVP-based power iteration, Hutchinson trace estimates, and spectral density routines.
* `spectral/hessian_evaluator.py` reloads saved weights + metadata, rebuilds the exact iterator order, and feeds batches into the Hessian routines for any stored run.

## Spectral Data CLI (`scripts/gather_hessian_spectra.py`)
* Command-line tool that lists available snapshot steps and runs `HessianEvaluator` on selected `(B, η, step)` combinations.
* Stores eigenvalues/trace in `spectral_data/<experiment_type>/<experiment>_spectra.pkl`, mirroring the experiment serialization layout.

## Integration & Tests
* Coverage for checkpoints/metadata, iterator behavior, sweep orchestration, Hessian evaluation, and linear/noisy teacher configs.
* Registry tests ensure spec builders stay valid and names remain unique.
* Dedicated smoke suite (`tests/smoke/`, `pytest -m smoke`) exercises the CLI and a tiny synthetic sweep; wired into pre-commit for quick regression checks.

---

In practice: define the experiment dataclass, register it, and use `scripts/run_experiments.py run` to launch a sweep. Use `scripts/run_experiments.py list` to see all registered experiments.
