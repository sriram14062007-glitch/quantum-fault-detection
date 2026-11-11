import unittest

import torch
import torch.nn as nn

from src.model import HybridModel
from src.quantum_layer import create_quantum_layer


class TestHybridPipeline(unittest.TestCase):
    def test_model_with_identity_quantum_layer(self):
        model = HybridModel(
            input_size=12,
            qlayer=nn.Identity(),
            n_qubits=6,
            hidden_dim=32,
            dropout=0.1,
            activation="relu",
        )
        output = model(torch.randn(4, 12))
        self.assertEqual(tuple(output.shape), (4, 2))

    def test_model_rejects_unknown_activation(self):
        with self.assertRaises(ValueError):
            HybridModel(12, nn.Identity(), 6, activation="unknown")

    def test_real_pennylane_layer_shapes(self):
        qlayer = create_quantum_layer(n_qubits=6, n_layers=6)
        weights = dict(qlayer.named_parameters())
        self.assertIn("weights", weights)
        self.assertEqual(tuple(weights["weights"].shape), (6, 6, 3))

        q_output = qlayer(torch.randn(2, 6))
        self.assertEqual(tuple(q_output.shape), (2, 6))

        model = HybridModel(12, qlayer, 6, hidden_dim=32, dropout=0.1)
        output = model(torch.randn(2, 12))
        self.assertEqual(tuple(output.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
