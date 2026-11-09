
import pennylane as qml


N_LAYERS = 6


def create_quantum_layer(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))

        qml.StronglyEntanglingLayers(
            weights,
            wires=range(n_qubits)
        )

        return [
            qml.expval(qml.PauliZ(i))
            for i in range(n_qubits)
        ]

    weight_shapes = {
        "weights": (N_LAYERS, n_qubits, 3)
    }

    return qml.qnn.TorchLayer(circuit, weight_shapes)
