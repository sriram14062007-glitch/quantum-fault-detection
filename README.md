# Quantum-Classical Hybrid Neural Network for Smart Grid Stability Classification

A research implementation of a hybrid quantum-classical neural network for binary smart-grid stability classification, evaluated against Random Forest and XGBoost baselines.

## Overview

Smart-grid stability is an important problem in modern power systems. Accurate classification of stable and unstable operating conditions can support monitoring and decision-making in electrical grids.

This project investigates a hybrid quantum-classical machine learning approach using a classical neural network combined with a parameterized quantum circuit.

The proposed model is evaluated against classical machine learning baselines using the Electrical Grid Stability Simulated Dataset.

**Research objective:** Evaluate the predictive performance, computational cost, and seed-to-seed variability of a hybrid quantum-classical classifier against classical baselines.

## Key Features

- Hybrid quantum-classical neural network using PennyLane and PyTorch.
- Parameterized quantum circuit with Angle Embedding and Strongly Entangling Layers.
- Random Forest and XGBoost baseline implementations.
- Stratified training, validation, and test splits.
- Training-only feature scaling to prevent data leakage.
- Multiple evaluation metrics and visualizations.
- Five-seed reproducibility experiments.
- Saved model checkpoints and experiment summaries.

## Dataset

This project uses the **Electrical Grid Stability Simulated Dataset** from the UCI Machine Learning Repository.

- Dataset source: https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data
- DOI: https://doi.org/10.24432/C5PG66
- Original creator: Vadim Arzamasov (2018)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset contains 10,000 simulated observations describing electrical grid operating conditions.

The target variable is `stabf`, which represents binary grid stability:

- `stable`
- `unstable`

### Dataset Attribution

This project uses the Electrical Grid Stability Simulated Data dataset provided by the UCI Machine Learning Repository.

- **Original creator:** Vadim Arzamasov (2018)
- **Source:** https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data
- **DOI:** https://doi.org/10.24432/C5PG66
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

The dataset is redistributed under its original license. All rights remain with the original creator.

The derived `stab` column is excluded from the input features to prevent target leakage.

### Data preprocessing

The active dataset used by the project is:

`data/sg_raw