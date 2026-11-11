from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_data(
    filepath="data/sg_raw_data.csv",
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    batch_size=64,
):
    """
    Load the UCI grid-stability CSV and return stratified train/validation/test
    loaders plus fitted preprocessing objects.

    `val_size` and `test_size` are fractions of the complete dataset.
    The scaler is fitted only on the training partition.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath.resolve()}. "
            "Place sg_raw_data.csv in the repository's data/ directory "
            "or pass --data-path."
        )

    if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1):
        raise ValueError("test_size and val_size must be positive and sum to less than 1.")

    df = pd.read_csv(filepath)
    required = {"stabf"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {sorted(missing)}")

    # 'stab' is derived from the target and must never be an input feature.
    if "stab" in df.columns:
        df = df.drop(columns=["stab"])

    X_df = df.drop(columns=["stabf"])
    if X_df.empty or not all(pd.api.types.is_numeric_dtype(t) for t in X_df.dtypes):
        raise ValueError("All input features must be numeric and at least one is required.")

    if X_df.isnull().any().any() or df["stabf"].isnull().any():
        raise ValueError("Dataset contains missing values. Clean/impute them explicitly first.")

    X = X_df.to_numpy(dtype=np.float32)
    y_raw = df["stabf"].to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw).astype(np.int64)
    if len(encoder.classes_) != 2:
        raise ValueError(f"Expected binary target with 2 classes; found {len(encoder.classes_)}.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    def make_loader(features, labels, shuffle):
        ds = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(labels.astype(np.int64)),
        )
        generator = torch.Generator()
        generator.manual_seed(random_state)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=0,
        )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "train_loader": make_loader(X_train, y_train, True),
        "val_loader": make_loader(X_val, y_val, False),
        "test_loader": make_loader(X_test, y_test, False),
        "scaler": scaler,
        "label_encoder": encoder,
        "feature_names": list(X_df.columns),
        "input_dim": X.shape[1],
    }


if __name__ == "__main__":
    data = load_and_preprocess_data()
    print("Data loaded successfully.")
    print(
        f"Train={len(data['X_train'])}, "
        f"Validation={len(data['X_val'])}, "
        f"Test={len(data['X_test'])}, "
        f"Features={data['input_dim']}"
    )
    print("Classes:", list(data["label_encoder"].classes_))
