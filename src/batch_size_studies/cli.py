"""
Command-line helpers for experiment sweeps.

This module owns argparse wiring so that runner.py can focus on orchestration.
"""

import argparse
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace

from .configs import get_main_experiment_configs, get_main_hyperparameter_grids
from .definitions import LossType, OptimizerType, Parameterization, RunKey
from .paths import EXPERIMENTS_DIR
from .runner import (
    _all_runs_accounted_for,
    _is_run_result_complete,
    _run_single_experiment,
    run_experiment_sweep,
)


@dataclass
class RunTask:
    name: str
    config: any


@dataclass
class RunPlan:
    directory: str
    batch_sizes: list[int]
    etas: list[float]
    tasks: list[RunTask]
    no_save: bool
    eta_stability_depth: int | None
    max_eval_samples: int | None
    save_interstitial_snapshots: bool | None
    save_epoch_snapshots: bool | None
    num_processes: int
    dry_run_task: RunTask | None = None
    dry_run_batch: int | None = None
    dry_run_eta: float | None = None
    dry_run_steps: int = 0


class RunPlanner:
    def __init__(self, args: argparse.Namespace, experiments: dict, batch_sizes, etas, directory):
        self.args = args
        self.experiments = experiments
        self.batch_sizes = batch_sizes
        self.etas = etas
        self.directory = directory

    def build(self) -> RunPlan | None:
        experiments = self.experiments
        if not experiments:
            logging.error("No experiments match the provided filters. Nothing to run.")
            return None

        if getattr(self.args, "override", None):
            experiments = _apply_overrides(experiments, self.args.override)

        if getattr(self.args, "dry_run", False):
            first_item = next(iter(experiments.items()), None)
            if first_item is None:
                logging.error("No experiments available for dry-run.")
                return None
            first_name, first_config = first_item
            dry_batch = min(self.batch_sizes) if self.batch_sizes else 1
            mid_eta = self.etas[len(self.etas) // 2] if self.etas else 0.1
            dry_task = RunTask(first_name, first_config)
            return RunPlan(
                directory=self.directory,
                batch_sizes=self.batch_sizes,
                etas=self.etas,
                tasks=[],
                no_save=True,
                eta_stability_depth=None,
                max_eval_samples=self.args.max_eval_samples,
                save_interstitial_snapshots=None,
                save_epoch_snapshots=None,
                num_processes=1,
                dry_run_task=dry_task,
                dry_run_batch=dry_batch,
                dry_run_eta=mid_eta,
                dry_run_steps=getattr(self.args, "dry_run_steps", 5),
            )

        filepaths = defaultdict(list)
        experiments_that_need_running = {}
        logging.info("--- Pre-flight check: Verifying experiments ---")

        for name, config in experiments.items():
            filepath = config.get_filepath(directory=self.directory)
            filepaths[filepath].append(name)
            if self.args.no_save:
                experiments_that_need_running[name] = config
            else:
                losses, failed = config.load_results(directory=self.directory, silent=True)
                if _all_runs_accounted_for(config, self.batch_sizes, self.etas, losses, failed):
                    logging.info(f"  Skipping '{name}': Already complete. (Found file: {os.path.basename(filepath)})")
                else:
                    logging.info(f"  Incomplete: '{name}'. Will run. (Checking file: {os.path.basename(filepath)})")
                    experiments_that_need_running[name] = config

        has_collision = False
        for filepath, names in filepaths.items():
            if len(names) > 1:
                logging.error(f"Collision detected! Experiments {names} will write to the same file: {filepath}")
                has_collision = True

        if has_collision:
            logging.error("\nAborting due to filename collisions.")
            return None

        if self.args.no_save:
            logging.info("\n--- --no-save enabled: All selected experiments will be run without saving. ---")

        if not experiments_that_need_running:
            logging.info("\n--- All experiments are already complete. Nothing to do. ---")
            return None

        tasks = [RunTask(name, config) for name, config in experiments_that_need_running.items()]
        return RunPlan(
            directory=self.directory,
            batch_sizes=self.batch_sizes,
            etas=self.etas,
            tasks=tasks,
            no_save=self.args.no_save,
            eta_stability_depth=self.args.eta_stability_depth,
            max_eval_samples=self.args.max_eval_samples,
            save_interstitial_snapshots=self.args.save_interstitial_snapshots,
            save_epoch_snapshots=self.args.save_epoch_snapshots,
            num_processes=self.args.num_processes,
        )


class RunExecutor:
    def run_dry(self, plan: RunPlan):
        task = plan.dry_run_task
        logging.info(
            "Dry-run mode: selecting the first experiment and a single (B, η) pair.\n"
            "Dry-run for '%s' @ (B=%s, η=%s) for %s steps.",
            task.name,
            plan.dry_run_batch,
            plan.dry_run_eta,
            plan.dry_run_steps,
        )
        run_experiment_sweep(
            experiment=task.config,
            batch_sizes=[plan.dry_run_batch],
            etas=[plan.dry_run_eta],
            init_key=0,
            directory=plan.directory,
            dry_run=True,
            dry_run_steps=plan.dry_run_steps,
            no_save=True,
            eta_stability_search_depth=None,
            max_eval_samples=plan.max_eval_samples,
        )

    def execute(self, plan: RunPlan):
        logging.info(f"\n--- Starting Pipeline for {len(plan.tasks)} Incomplete Experiments ---")
        if plan.num_processes <= 1:
            logging.info("Running experiments sequentially.")
            for task in plan.tasks:
                try:
                    _run_single_experiment(
                        task.name,
                        task.config,
                        plan.batch_sizes,
                        plan.etas,
                        plan.directory,
                        plan.no_save,
                        plan.eta_stability_depth,
                        plan.max_eval_samples,
                        plan.save_interstitial_snapshots,
                        plan.save_epoch_snapshots,
                    )
                except Exception as exc:  # pragma: no cover - logged for visibility
                    logging.error(f"Experiment '{task.name}' generated an exception: {exc}")
        else:
            logging.info(f"Running experiments with up to {plan.num_processes} parallel workers.")
            with ProcessPoolExecutor(max_workers=plan.num_processes) as executor:
                future_to_name = {
                    executor.submit(
                        _run_single_experiment,
                        task.name,
                        task.config,
                        plan.batch_sizes,
                        plan.etas,
                        plan.directory,
                        plan.no_save,
                        plan.eta_stability_depth,
                        plan.max_eval_samples,
                        plan.save_interstitial_snapshots,
                        plan.save_epoch_snapshots,
                    ): task.name
                    for task in plan.tasks
                }

                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        future.result()
                    except Exception as exc:  # pragma: no cover - logged for visibility
                        logging.error(f"Experiment '{name}' generated an exception: {exc}")


def _coerce_enum(parser, enum_cls, raw_value, flag_name):
    if raw_value is None:
        return None
    candidate = raw_value.strip().lower()
    for member in enum_cls:
        if member.name.lower() == candidate or str(member.value).lower() == candidate:
            return member
    valid = ", ".join([f"{m.name} ({m.value})" for m in enum_cls])
    parser.error(f"Invalid value '{raw_value}' for {flag_name}. Valid choices: {valid}.")


def add_filter_args(parser: argparse.ArgumentParser):
    """Adds shared experiment filtering arguments to a parser."""
    parser.add_argument(
        "-n",
        "--name",
        nargs="*",
        help="Filter by the specific experiment name(s).",
    )
    parser.add_argument(
        "--experiment-type",
        action="append",
        dest="experiment_types",
        help="Filter experiments by their experiment_type string. Repeat the flag to include multiple types.",
    )
    parser.add_argument(
        "--optimizer",
        "--opt",
        dest="optimizer",
        help="Filter experiments by optimizer (e.g., SGD, Adam). Case-insensitive.",
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        help="Filter experiments by loss function (e.g., MSE, XENT). Case-insensitive.",
    )
    parser.add_argument(
        "--list-overrides",
        action="store_true",
        help="Show supported override keys (for -o KEY=VALUE) and exit.",
    )


def describe_supported_overrides() -> str:
    return "\n".join(
        [
            "Supported override keys:",
            "  - num_epochs=<int>: force a fixed number of epochs for fixed-data experiments.",
            "  - max_eval_samples=<int>: cap evaluation set size per epoch.",
            "  - save_interstitial_snapshots=<bool>: enable/disable dense weight snapshots.",
            "  - save_epoch_snapshots=<bool>: toggle per-epoch snapshots for fixed-data synthetic runs.",
            "  - disable_eval_dataset=<bool>: skip deterministic synthetic eval dataset (saves memory).",
            "  - dataset_loader=<callable>: alternate loader for MNIST-style datasets.",
            "  - forced_subsample_seed=<int>: deterministic seed for subsampled datasets.",
        ]
    )


def _resolve_experiment_configs(args: argparse.Namespace):
    optimizer_filter = _coerce_enum(argparse.ArgumentParser(), OptimizerType, args.optimizer, "--optimizer")
    loss_filter = _coerce_enum(argparse.ArgumentParser(), LossType, args.loss, "--loss")
    config_kwargs = {}
    if getattr(args, "experiment_types", None):
        config_kwargs["experiment_types"] = args.experiment_types
    if optimizer_filter is not None:
        config_kwargs["optimizer"] = optimizer_filter
    if loss_filter is not None:
        config_kwargs["loss_type"] = loss_filter

    experiments = get_main_experiment_configs(**config_kwargs)
    if args.name:
        experiments = {name: config for name, config in experiments.items() if name in args.name}
        if not experiments:
            logging.error(f"No experiments found with name(s): {args.name}. Aborting.")
            return None
    return experiments


def _handle_list_command(args: argparse.Namespace, experiments_to_run: dict) -> None:
    if getattr(args, "list_overrides", False):
        print(describe_supported_overrides())
        return

    logging.info("--- Available Experiments ---")
    headers = ["NAME", "TYPE", "OPTIMIZER", "LOSS"]
    rows = [
        [name, config.experiment_type, config.optimizer.name, config.loss_type.name]
        for name, config in experiments_to_run.items()
    ]

    if not rows:
        logging.info("No experiments match the provided filters.")
        return

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "=" * len(header_line)
    print(f"\n{separator}")
    print(f"Available Experiments ({len(rows)} total)")
    print(f"{separator}")
    print(header_line)
    print("-" * len(header_line))

    for row in sorted(rows):
        row_line = "  ".join(c.ljust(w) for c, w in zip(row, col_widths))
        print(row_line)

    print(f"{separator}\n")


def _apply_overrides(experiments_to_run: dict, overrides_list: list[str]) -> dict:
    overrides = {}
    for override_str in overrides_list:
        key, value_str = override_str.split("=", 1)
        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                if key == "parameterization":
                    value = Parameterization[value_str.upper()]
                else:
                    value = value_str
        overrides[key] = value

    logging.info(f"Applying overrides: {overrides}")
    return {name: replace(config, **overrides) for name, config in experiments_to_run.items()}


def _handle_run_command(args: argparse.Namespace, experiments_to_run: dict) -> None:
    directory = EXPERIMENTS_DIR
    batch_sizes, etas = get_main_hyperparameter_grids()
    planner = RunPlanner(args, experiments_to_run, batch_sizes, etas, directory)
    plan = planner.build()
    if plan is None:
        return

    executor = RunExecutor()
    if plan.dry_run_task:
        executor.run_dry(plan)
        return

    executor.execute(plan)
    logging.info("\n--- All experiments complete. ---")


def _extract_summary_metric(result: dict) -> tuple[float | None, str]:
    """
    Returns a scalar metric and a label indicating whether higher is better ("acc") or lower is better ("loss").
    """
    if not isinstance(result, dict):
        return None, "loss"
    if (acc := result.get("final_test_accuracy")) is not None:
        return float(acc), "acc"
    if epoch_accs := result.get("epoch_test_accuracies"):
        return float(epoch_accs[-1]), "acc"
    if (eval_loss := result.get("final_eval_loss")) is not None:
        return float(eval_loss), "loss"
    if history := result.get("loss_history"):
        return float(history[-1]), "loss"
    return None, "loss"


def _handle_summary_command(args: argparse.Namespace, experiments_to_run: dict) -> None:
    directory = EXPERIMENTS_DIR
    batch_sizes, etas = get_main_hyperparameter_grids()

    for name, config in experiments_to_run.items():
        results_dict, failed_runs = config.load_results(directory=directory, silent=True)
        total_candidates = 0
        complete = 0
        failed = len(failed_runs)
        best_loss = None
        best_loss_key = None
        best_acc = None
        best_acc_key = None

        for batch_size in batch_sizes:
            if config.should_skip_batch_size(batch_size, train_ds=None):
                continue
            for eta in etas:
                total_candidates += 1
                run_key = RunKey(batch_size=batch_size, eta=eta)
                result = results_dict.get(run_key)
                if _is_run_result_complete(result):
                    complete += 1
                    metric, kind = _extract_summary_metric(result)
                    if metric is not None:
                        if kind == "loss":
                            if best_loss is None or metric < best_loss:
                                best_loss = metric
                                best_loss_key = run_key
                        else:
                            if best_acc is None or metric > best_acc:
                                best_acc = metric
                                best_acc_key = run_key
                elif run_key in failed_runs:
                    continue

        missing = max(total_candidates - complete - failed, 0)

        logging.info("=== %s ===", name)
        logging.info(
            "Total grid: %s, complete: %s, failed: %s, missing: %s", total_candidates, complete, failed, missing
        )
        if best_loss is not None:
            logging.info("Best loss: %.4g @ %s", best_loss, best_loss_key)
        if best_acc is not None:
            logging.info("Best accuracy: %.4g @ %s", best_acc, best_acc_key)
        if best_loss is None and best_acc is None:
            logging.info("No completed runs yet.")


def run_from_cli_args(args: argparse.Namespace):
    experiments_to_run = _resolve_experiment_configs(args)
    if experiments_to_run is None:
        return

    match args.command:
        case "list":
            _handle_list_command(args, experiments_to_run)
        case "run":
            _handle_run_command(args, experiments_to_run)
        case "summary":
            _handle_summary_command(args, experiments_to_run)
        case _:
            logging.error(f"Unknown command: {args.command}")
