import argparse
import json
import random
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from src.dataset import load_and_preprocess_data
from src.quantum_layer import create_quantum_layer
from src.model import HybridModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
            all_targets.extend(batch_y.detach().cpu().numpy())

    mean_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    return mean_loss, accuracy, np.asarray(all_targets), np.asarray(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Train hybrid quantum-classical grid fault classifier.")
    parser.add_argument("--data-path", default="data/sg_raw_data.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", choices=["relu", "gelu", "tanh", "leaky_relu"], default="relu")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive.")

    set_seed(args.seed)
    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 58)
    print("Hybrid Quantum-Classical Smart Grid Fault Detection")
    print("=" * 58)
    print("Device:", device)

    data = load_and_preprocess_data(
        filepath=args.data_path,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    print(
        f"Samples: train={len(data['X_train'])}, "
        f"validation={len(data['X_val'])}, test={len(data['X_test'])}; "
        f"features={data['input_dim']}"
    )

    qlayer = create_quantum_layer(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
    )
    model = HybridModel(
        input_size=data["input_dim"],
        qlayer=qlayer,
        n_qubits=args.n_qubits,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        activation=args.activation,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_path = output_dir / "best_model.pt"

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model, data["train_loader"], criterion, device, optimizer
        )
        val_loss, val_acc, _, _ = run_epoch(
            model, data["val_loader"], criterion, device
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "input_dim": data["input_dim"],
                    "feature_names": data["feature_names"],
                    "classes": data["label_encoder"].classes_.tolist(),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss={train_loss:.4f}, acc={train_acc:.4f} | "
                f"val loss={val_loss:.4f}, acc={val_acc:.4f}"
            )

    elapsed = time.time() - start
    print(f"Training time: {elapsed:.2f} seconds; best epoch: {best_epoch}")

    # Reload the checkpoint selected only using validation loss.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_targets, test_preds, test_probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in data["test_loader"]:
            logits = model(batch_x.to(device))
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)
            test_targets.extend(batch_y.numpy())
            test_preds.extend(predictions.cpu().numpy())
            test_probs.extend(probabilities.cpu().numpy())

    test_targets = np.asarray(test_targets)
    test_preds = np.asarray(test_preds)
    test_probs = np.asarray(test_probs)

    metrics = {
        "accuracy": float(accuracy_score(test_targets, test_preds)),
        "precision": float(precision_score(test_targets, test_preds, zero_division=0)),
        "recall": float(recall_score(test_targets, test_preds, zero_division=0)),
        "f1_score": float(f1_score(test_targets, test_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_targets, test_probs)),
        "confusion_matrix": confusion_matrix(test_targets, test_preds, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            test_targets,
            test_preds,
            labels=[0, 1],
            target_names=[str(x) for x in data["label_encoder"].classes_],
            output_dict=True,
            zero_division=0,
        ),
    }

    report = {
        "metrics": metrics,
        "experiment": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "n_qubits": args.n_qubits,
            "n_layers": args.n_layers,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "activation": args.activation,
            "train_samples": len(data["X_train"]),
            "validation_samples": len(data["X_val"]),
            "test_samples": len(data["X_test"]),
            "feature_names": data["feature_names"],
            "class_names": [str(x) for x in data["label_encoder"].classes_],
            "training_seconds": elapsed,
            "device": str(device),
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    joblib.dump(data["scaler"], output_dir / "scaler.joblib")
    joblib.dump(data["label_encoder"], output_dir / "label_encoder.joblib")
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    # Confusion matrix
    cm = np.asarray(metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=data["label_encoder"].classes_,
        yticklabels=data["label_encoder"].classes_,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(test_targets, test_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Test ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close()

    print("\nFinal held-out test metrics")
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"{key:>12}: {metrics[key]:.4f}")
    print("Confusion matrix:\n", cm)
    print(f"\nArtifacts saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
