import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, input_size, qlayer, n_qubits):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.fc_mid = nn.Linear(8, n_qubits)
        self.q_layer = qlayer
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc_mid(x)
        x = self.q_layer(x)
        x = self.fc2(x)
        return x