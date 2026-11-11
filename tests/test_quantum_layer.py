
import torch

from src.quantum_layer import create_quantum_layer
from src.model import HybridModel

# Configure quantum layer
n_qubits = 6
qlayer = create_quantum_layer(n_qubits)

# Verify trainable quantum parameters
for name, param in qlayer.named_parameters():
    print(name, param.shape)

# Test quantum layer independently
x = torch.randn(2, n_qubits)
quantum_output = qlayer(x)

print("Quantum output:", quantum_output.shape)

# Test complete hybrid model
model = HybridModel(
    input_size=12,
    qlayer=qlayer,
    n_qubits=n_qubits
)

features = torch.randn(2, 12)
output = model(features)

print("Hybrid output:", output.shape)

assert quantum_output.shape == (2, 6)
assert output.shape == (2, 2)

print("All quantum layer tests passed!")
