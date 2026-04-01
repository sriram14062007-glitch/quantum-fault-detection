import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Store losses
losses = []

# Training loop
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    # Store loss
    losses.append(loss.item())

    # Print every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# ===== AFTER TRAINING ===== #

# Predictions (Train)
train_preds = torch.argmax(model(X_train_tensor), axis=1).detach().numpy()

# Predictions (Test)
test_outputs = model(X_test_tensor)
test_preds = torch.argmax(test_outputs, axis=1).detach().numpy()

# Accuracy
print("\nFinal Results:")
print("Train Accuracy:", accuracy_score(y_train, train_preds))
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Training script placeholder")