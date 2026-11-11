from pathlib import Path
import argparse
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from src.dataset import load_and_preprocess_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost baseline for smart-grid fault detection."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", default="data/sg_raw_data.csv")
    parser.add_argument("--output-dir", default="results/xgboost")
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    early_stopping_rounds = args.early_stopping_rounds

    random.seed(seed)
    np.random.seed(seed)

    data = load_and_preprocess_data(
        filepath=args.data_path,
        random_state=seed,
    )

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    class_names = [str(c) for c in data["label_encoder"].classes_]

    print("=" * 55)
    print("XGBoost - Smart Grid Fault Detection")
    print("=" * 55)
    print("XGBoost version:", xgb.__version__)
    print("Random seed:", seed)
    print("Classes:", class_names)
    print(
        f"Samples: train={len(X_train)}, "
        f"validation={len(X_val)}, test={len(X_test)}"
    )

    params = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    model = XGBClassifier(**params)
    start = time.perf_counter()

    # XGBoost 2.x/3.x accepts early_stopping_rounds in estimator params.
    # Keep a fallback for older versions.
    try:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    except TypeError:
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

    training_time = time.perf_counter() - start
    best_iteration = getattr(model, "best_iteration", None)

    print("\nBest iteration:", best_iteration)

    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]

    print("\nValidation results")
    print("Accuracy:", accuracy_score(y_val, val_pred))
    print("F1-score:", f1_score(y_val, val_pred, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_val, val_prob))

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, test_pred)

    metrics = {
        "model": "XGBoost",
        "xgboost_version": xgb.__version__,
        "random_seed": seed,
        "parameters": params,
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": (
            int(best_iteration) if best_iteration is not None else None
        ),
        "training_time_seconds": training_time,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "classes": class_names,
        "validation_accuracy": accuracy_score(y_val, val_pred),
        "validation_f1_score": f1_score(y_val, val_pred, zero_division=0),
        "validation_roc_auc": roc_auc_score(y_val, val_prob),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_precision": precision_score(y_test, test_pred, zero_division=0),
        "test_recall": recall_score(y_test, test_pred, zero_division=0),
        "test_f1_score": f1_score(y_test, test_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, test_prob),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            test_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    model.save_model(str(output_dir / "model.json"))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("XGBoost - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, test_prob)
    plt.title("XGBoost - Test ROC Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close()

    importance = model.feature_importances_
    indices = np.argsort(importance)
    plt.figure(figsize=(9, 6))
    plt.barh(
        np.array(data["feature_names"])[indices],
        importance[indices],
    )
    plt.xlabel("Feature importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close()

    print("\nFinal held-out test metrics")
    for key in (
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1_score",
        "test_roc_auc",
    ):
        print(f"{key}: {metrics[key]:.4f}")

    print("\nConfusion matrix:")
    print(cm)
    print(f"\nTraining time: {training_time:.2f} seconds")
    print("Artifacts saved in:", output_dir.resolve())


if __name__ == "__main__":
    main()
