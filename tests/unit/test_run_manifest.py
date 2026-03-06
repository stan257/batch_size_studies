import json
from pathlib import Path

from batch_size_studies.io.run_manifest import (
    SCHEMA_VERSION,
    build_sweep_manifest_payload,
    load_run_manifest,
    manifest_path_from_weights,
    save_run_manifest,
)


def test_manifest_path_from_weights():
    weights_path = "/tmp/example_weights.pkl"
    manifest_path = manifest_path_from_weights(weights_path)
    assert manifest_path.endswith("_manifest.json")
    assert manifest_path == "/tmp/example_manifest.json"


def test_save_and_load_run_manifest_merges_payload(tmp_path):
    weights_path = str(tmp_path / "demo_weights.pkl")
    payload1 = {"status": "running", "sweep": {"init_key": 0}}
    payload2 = {"status": "completed", "provenance": {"git_commit": "abc123"}}

    save_run_manifest(weights_path, payload1, merge=True)
    save_run_manifest(weights_path, payload2, merge=True)

    loaded = load_run_manifest(weights_path)
    assert loaded["status"] == "completed"
    assert loaded["sweep"]["init_key"] == 0
    assert loaded["provenance"]["git_commit"] == "abc123"

    manifest_file = Path(manifest_path_from_weights(weights_path))
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw == loaded


def test_build_sweep_manifest_payload_contains_schema():
    payload = build_sweep_manifest_payload(
        experiment_type="mnist1m_classification",
        experiment_params={"N": 128, "optimizer": "SGD"},
        batch_sizes=[1, 2, 4],
        etas=[0.1, 0.2],
        init_key=0,
        status="running",
        run_options={"max_eval_samples": 128},
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"] == "running"
    assert payload["experiment"]["type"] == "mnist1m_classification"
    assert payload["sweep"]["batch_sizes"] == [1, 2, 4]
