import numpy as np


# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Initialize parameters
w = np.random.randn(1)  # weight
b = np.random.randn(1)  # bias

# Input features and true labels
X = np.array([1])
y = np.array([1])

# Hyperparameters
learning_rate = 0.1
num_epochs = 1000

# Optimization loop
for epoch in range(num_epochs):
    # Forward pass
    z = np.dot(X, w) + b
    p = sigmoid(z)

    # Compute Binary Cross-Entropy Loss
    loss = - (y * np.log(p) + (1 - y) * np.log(1 - p))

    # Backpropagation
    dz = p - y
    dw = np.dot(X.T, dz)
    db = np.sum(dz)

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    # Print loss
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Loss = {loss}')

# Print optimized parameters
print(f'Optimized parameters: w = {w}, b = {b}')