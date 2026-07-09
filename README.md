# Quantum-Classical Hybrid Neural Network for Power Grid Fault Detection

This project uses a mix of quantum and classical machine learning to decide whether the power grid is stable or faulty, built using **PennyLane** and **PyTorch**.

## Problem Statement

If we miss early signs of trouble in the power grid, small issues can snowball into massive blackouts. In this project, we ask whether a **quantum‑enhanced neural network** which blends regular dense layers with a variational quantum circuit—can reliably tell apart healthy grid states from faulty ones. We then pit this hybrid model against a purely classical baseline to see whether the extra quantum machinery actually earns its place by improving performance.

## Dataset

[Electrical Grid Stability Simulated Dataset](https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data) (UCI) — 12 predictive features (reaction times `tau1-4`, power balance `p1-4`, price elasticity coefficients `g1-4`) used to classify grid state as **stable** or **unstable**.

## Architecture

- **Classical layers:** Dense (fully connected) layers for initial feature extraction and dimensionality reduction
- **Quantum layer:** Variational quantum circuit (PennyLane) using angle embedding to encode classical features into qubit rotations, followed by entangling gates
- **Output layer:** Classical dense layer for final binary classification (fault / no fault)

 *(results/architecture.png)`)*

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| Random Forest (classical baseline) | **92.4%** | Same preprocessed features, no target leakage |
| Hybrid Quantum-Classical NN (initial) | 84.6% | 6 qubits, 3 entangling layers, 100 epochs |
| **Hybrid Quantum-Classical NN (tuned)** | **91.0%** | 6 qubits, 6 entangling layers, fixed output layer size, 100 epochs |

*Note: an earlier version of this project reported 98–100% accuracy for both models due to a target-leakage bug (the continuous `stab` column, from which the `stabf` label is directly derived, was accidentally left in the feature set). After fixing the leak, correcting the scaler to fit only on training data, and tuning the quantum model, these are the corrected, reproducible results.*

**What closed the gap (84.6% → 91.0%):**
1. Fixed the output layer size (`nn.Linear(n_qubits, 3)` → `nn.Linear(n_qubits, 2)`) — the original had 3 output classes for what is a binary (stable/unstable) problem
2. Increased `StronglyEntanglingLayers` depth from 3 to 6, giving the circuit more trainable parameters
3. Added a fixed random seed (`torch.manual_seed(42)`) — without it, the quantum layer and classical layers get different random weight initializations on every run, causing test accuracy to vary by several points run-to-run (observed range: ~85–91%). With the seed set, results are reproducible.

The tuned quantum hybrid model now performs close to the classical Random Forest baseline, showing that with correct architecture and sufficient circuit depth, the hybrid approach is competitive on this task — even though the classical model retains a small edge.

**Confusion Matrix & Loss Curve:** see `results/`

## Tech Stack

`Python` · `PyTorch` · `PennyLane` · `Scikit-learn` · `NumPy` · `Matplotlib`

## Project Structure

```
quantum-fault-detection/
├── data/           # Dataset(s) used for training/testing
├── notebooks/      # Jupyter notebooks (EDA, training, evaluation)
├── results/        # Plots: loss curves, confusion matrix, architecture diagram
├── src/            # Core model and utility scripts
├── requirements.txt
└── README.md
```

## How to Run

```bash
# Clone the repo
git clone https://github.com/sriram14062007-glitch/quantum-fault-detection.git
cd quantum-fault-detection

# Install dependencies
pip install -r requirements.txt

# Run the main notebook
jupyter notebook notebooks/<main_notebook>.ipynb
```

## Key Learnings

- Practical experience combining **quantum computing (PennyLane)** with classical deep learning (PyTorch) in a single trainable pipeline
- Hands-on work with variational quantum circuits, angle embedding, and entanglement for feature encoding
- End-to-end ML workflow: data preprocessing → model design → training → evaluation → benchmarking against a classical baseline

## Future Work

- Try 8+ qubits (needs more compute than a free Colab CPU runtime comfortably handles) to see if the remaining gap to Random Forest closes further
- Add precision/recall/F1 and cross-validation for a more complete evaluation
- Explore noise-resilience for deployment on real quantum hardware (IBM Q, AWS Braket)
- Compare against additional classical models (XGBoost, SVM, Gradient Boosting)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
