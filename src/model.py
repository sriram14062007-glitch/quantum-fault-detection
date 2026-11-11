import torch
import torch.nn as nn


class HybridModel(nn.Module):
    """Classical feature projection + PennyLane TorchLayer + binary logits."""

    def __init__(
        self,
        input_size,
        qlayer,
        n_qubits,
        hidden_dim=32,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        if input_size <= 0 or n_qubits <= 0 or hidden_dim <= 0:
            raise ValueError("input_size, n_qubits and hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }
        activation_key = activation.lower()
        if activation_key not in activations:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {sorted(activations)}."
            )

        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.activation = activations[activation_key]()
        self.dropout = nn.Dropout(dropout)
        self.fc_mid = nn.Linear(hidden_dim, n_qubits)
        self.q_layer = qlayer
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_mid(x)
        x = self.q_layer(x)
        return self.fc2(x)
