"""Sidecar run manifest utilities.

These helpers write human-readable JSON metadata next to binary weight files
without changing the existing pickle schemas used for backward compatibility.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1


def manifest_path_from_weights(weights_filepath: str) -> str:
    if weights_filepath.endswith("_weights.pkl"):
        return weights_filepath[: -len("_weights.pkl")] + "_manifest.json"
    base, _ = os.path.splitext(weights_filepath)
    return base + "_manifest.json"


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_run_manifest(weights_filepath: str) -> dict[str, Any]:
    path = manifest_path_from_weights(weights_filepath)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_run_manifest(weights_filepath: str, payload: dict[str, Any], merge: bool = True) -> str:
    path = manifest_path_from_weights(weights_filepath)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = payload
    if merge:
        existing = load_run_manifest(weights_filepath)
        data = _merge_dicts(existing, payload)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)
    return path


def build_sweep_manifest_payload(
    *,
    experiment_type: str,
    experiment_params: dict[str, Any],
    batch_sizes: list[int],
    etas: list[float],
    init_key: int,
    status: str,
    run_options: dict[str, Any] | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "experiment": {
            "type": experiment_type,
            "params": experiment_params,
        },
        "sweep": {
            "batch_sizes": list(batch_sizes),
            "etas": list(etas),
            "init_key": int(init_key),
        },
    }
    if run_options:
        payload["sweep"]["options"] = run_options
    if git_commit is not None:
        payload["provenance"] = {"git_commit": git_commit}
    return payload
