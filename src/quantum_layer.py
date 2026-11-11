import pennylane as qml


def create_quantum_layer(
    n_qubits=6,
    n_layers=6,
    rotation="X",
    dev_name="default.qubit",
):
    """Create a trainable PennyLane quantum circuit wrapped as a TorchLayer."""
    if n_qubits <= 0 or n_layers <= 0:
        raise ValueError("n_qubits and n_layers must be positive.")
    if rotation not in {"X", "Y", "Z"}:
        raise ValueError("rotation must be one of 'X', 'Y' or 'Z'.")

    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)
