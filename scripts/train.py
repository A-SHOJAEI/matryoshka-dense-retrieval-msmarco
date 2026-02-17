#!/usr/bin/env python3
"""Pipeline-compatible training wrapper for matryoshka dense retrieval.

Handles:
- Data preparation (toy data for smoke test, MS MARCO for full training)
- Training both experiments (dense_baseline + matryoshka)
- Saving results in pipeline-expected format
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"


def run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=3600
    )


def main() -> None:
    config = "configs/smoke.yaml" if SMOKE_TEST else "configs/matryoshka.yaml"
    config_path = PROJECT_ROOT / config

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    python = sys.executable

    # Step 1: Prepare data
    print(f"=== Preparing data with config: {config} ===")
    result = run_cmd(
        [python, "-m", "src.data.make_data", "--config", config],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"Data preparation failed:\n{result.stderr}")
        sys.exit(1)
    print("Data preparation complete.")

    # Step 2: Train both experiments
    experiments = ["dense_baseline", "matryoshka"]
    for exp in experiments:
        print(f"\n=== Training experiment: {exp} ===")
        result = run_cmd(
            [python, "-m", "src.train.biencoder", "--config", config, "--experiment", exp],
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            print(f"Training {exp} failed:\n{result.stderr}")
            # Continue to next experiment rather than failing completely
            continue
        print(f"Training {exp} complete.")

    # Step 3: Run evaluation
    print("\n=== Running evaluation ===")
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    results_file = artifacts_dir / "results.json"

    result = run_cmd(
        [python, "-m", "src.eval.run_eval", "--config", config, "--out", str(results_file)],
        cwd=PROJECT_ROOT,
    )

    # Step 4: Save results in pipeline format
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    training_results = {"config": config, "experiments": experiments}

    if results_file.exists():
        try:
            eval_results = json.loads(results_file.read_text())
            training_results["test_metrics"] = eval_results
        except (json.JSONDecodeError, OSError):
            pass

    (results_dir / "training_results.yaml").write_text(
        json.dumps(training_results, indent=2, default=str) + "\n"
    )

    print(f"\n=== Training complete. Results saved to {results_dir} ===")


if __name__ == "__main__":
    main()
