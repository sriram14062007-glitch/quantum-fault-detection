"""Aggregate five-seed smart-grid fault-detection experiments.

Expected layout:
results/
  metrics.json                         # quantum, seed 42
  random_forest/metrics.json           # RF, seed 42
  xgboost/metrics.json                 # XGBoost, seed 42
  reproducibility/
    seed_7/{quantum,random_forest,xgboost}/metrics.json
    seed_21/{quantum,random_forest,xgboost}/metrics.json
    seed_100/{quantum,random_forest,xgboost}/metrics.json
    seed_2026/{quantum,random_forest,xgboost}/metrics.json

Run from the project root:
  python -m src.aggregate_reproducibility
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_DIR = RESULTS / "reproducibility" / "summary"

SEEDS = [42, 7, 21, 100, 2026]
MODELS = ["quantum", "random_forest", "xgboost"]

METRIC_KEYS = {
    "accuracy": ("accuracy", "test_accuracy"),
    "precision": ("precision", "test_precision"),
    "recall": ("recall", "test_recall"),
    "f1_score": ("f1_score", "test_f1_score"),
    "roc_auc": ("roc_auc", "test_roc_auc"),
}

def metrics_path(seed: int, model: str) -> Path:
    if seed == 42:
        if model == "quantum":
            return RESULTS / "metrics.json"
        return RESULTS / model / "metrics.json"
    return RESULTS / "reproducibility" / f"seed_{seed}" / model / "metrics.json"

def first_present(mapping: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None

def read_experiment(seed: int, model: str) -> dict:
    path = metrics_path(seed, model)

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing metrics file for seed={seed}, model={model}: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Quantum metrics are nested; classical metrics are at root.
    metric_block = raw.get("metrics", raw)

    row = {
        "seed": seed,
        "model": model,
        "source_file": str(path.relative_to(ROOT)),
    }

    for output_key, candidates in METRIC_KEYS.items():
        value = first_present(metric_block, candidates)

        if value is None and metric_block is not raw:
            value = first_present(raw, candidates)

        if value is None:
            raise ValueError(
                f"Missing metric '{output_key}' in {path}"
            )

        row[output_key] = float(value)

    # Read training time from root or nested experiment metadata.
    time_value = first_present(
        raw,
        ("training_seconds", "training_time_seconds", "training_time"),
    )

    if time_value is None:
        time_value = first_present(
            raw.get("experiment", {}),
            ("training_seconds", "training_time_seconds", "training_time"),
        )

    if time_value is None:
        raise ValueError(f"Missing training-time field in {path}")

    row["training_time_seconds"] = float(time_value)

    # Validate seed metadata.
    recorded_seed = raw.get("random_seed")

    if recorded_seed is None:
        recorded_seed = raw.get("experiment", {}).get("seed")

    if recorded_seed is not None and int(recorded_seed) != seed:
        raise ValueError(
            f"Seed mismatch in {path}: expected {seed}, found {recorded_seed}"
        )

    # Validate dataset split sizes.
    for field, expected in (
        ("train_samples", 7000),
        ("validation_samples", 1500),
        ("test_samples", 1500),
    ):
        actual = raw.get(field)

        if actual is None:
            actual = raw.get("experiment", {}).get(field)

        if actual is not None and int(actual) != expected:
            raise ValueError(
                f"{path}: expected {field}={expected}, found {actual}"
            )

    # Preserve confusion matrix.
    cm = metric_block.get(
        "confusion_matrix",
        raw.get("confusion_matrix"),
    )

    row["confusion_matrix"] = (
        json.dumps(cm, separators=(",", ":"))
        if cm is not None
        else ""
    )

    return row

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []
    for seed in SEEDS:
        for model in MODELS:
            try:
                rows.append(read_experiment(seed, model))
            except Exception as exc:
                errors.append(str(exc))

    if errors:
        print("Aggregation stopped. Please fix these issues:")
        for error in errors:
            print(f" - {error}")
        raise SystemExit(1)

    if len(rows) != len(SEEDS) * len(MODELS):
        raise SystemExit(f"Expected 15 experiment rows, found {len(rows)}")

    raw_csv = OUTPUT_DIR / "five_seed_all_runs.csv"
    columns = [
        "seed", "model", "accuracy", "precision", "recall",
        "f1_score", "roc_auc", "training_time_seconds",
        "confusion_matrix", "source_file",
    ]
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        for metric in (
            "accuracy", "precision", "recall", "f1_score",
            "roc_auc", "training_time_seconds",
        ):
            values = [r[metric] for r in model_rows]
            summary.append({
                "model": model,
                "metric": metric,
                "n_seeds": len(values),
                "mean": mean(values),
                "std_sample": stdev(values),
                "values_by_seed": {
                    str(r["seed"]): r[metric] for r in model_rows
                },
            })

    summary_json = OUTPUT_DIR / "five_seed_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({
            "seeds": SEEDS,
            "models": MODELS,
            "n_experiments": len(rows),
            "standard_deviation": "sample standard deviation (ddof=1)",
            "summary": summary,
        }, f, indent=2)

    summary_csv = OUTPUT_DIR / "five_seed_mean_std.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "metric", "n_seeds", "mean", "std_sample"]
        )
        writer.writeheader()
        for item in summary:
            writer.writerow({k: item[k] for k in writer.fieldnames})

    print("Aggregation successful.")
    print(f"Experiments verified: {len(rows)} / 15")
    print(f"Seeds: {SEEDS}")
    print(f"Models: {MODELS}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f" - {raw_csv.name}")
    print(f" - {summary_csv.name}")
    print(f" - {summary_json.name}")
    print("\nMean ± sample standard deviation:")
    for model in MODELS:
        print(f"\n{model}")
        for metric in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
            item = next(x for x in summary if x["model"] == model and x["metric"] == metric)
            print(
                f"  {metric:10s}: "
                f"{item['mean'] * 100:.3f} ± {item['std_sample'] * 100:.3f}%"
            )

if __name__ == "__main__":
    main()
