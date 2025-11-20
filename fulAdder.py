import numpy as np
import matplotlib.pyplot as plt

# Activation: Bipolar tanh
def bipolar_tanh(x):
    return np.tanh(x)

def bipolar_tanh_derivative(x):
    return 1 - np.tanh(x)**2


# Bipolar inputs for Full Adder (A, B, Cin)
inputs = np.array([
    [-1, -1, -1],  # 000
    [-1, -1,  1],  # 001
    [-1,  1, -1],  # 010
    [-1,  1,  1],  # 011
    [ 1, -1, -1],  # 100
    [ 1, -1,  1],  # 101
    [ 1,  1, -1],  # 110
    [ 1,  1,  1]   # 111
])

# Expected outputs (SUM, CARRY) in bipolar form (0 → -1, 1 → 1)
expected = np.array([
    [-1, -1],  # 000 → 0,0
    [ 1, -1],  # 001 → 1,0
    [ 1, -1],  # 010 → 1,0
    [-1,  1],  # 011 → 0,1
    [ 1, -1],  # 100 → 1,0
    [-1,  1],  # 101 → 0,1
    [-1,  1],  # 110 → 0,1
    [ 1,  1]   # 111 → 1,1
])

np.random.seed(42)
input_size = 3
hidden_size = 6      # full adder is more complex → bigger hidden layer
output_size = 2

# Initialize weights and biases
W1 = np.random.uniform(-1, 1, (hidden_size, input_size))
b1 = np.random.uniform(-1, 1, hidden_size)
W2 = np.random.uniform(-1, 1, (output_size, hidden_size))
b2 = np.random.uniform(-1, 1, output_size)

lr = 0.1
epochs = 20000


# Training loop
for epoch in range(epochs):
    for x, y_true in zip(inputs, expected):

        # Forward pass
        z1 = np.dot(W1, x) + b1
        a1 = bipolar_tanh(z1)

        z2 = np.dot(W2, a1) + b2
        a2 = bipolar_tanh(z2)

        # Error
        error = y_true - a2

        # Backpropagation
        delta2 = error * bipolar_tanh_derivative(z2)
        delta1 = bipolar_tanh_derivative(z1) * (W2.T @ delta2)

        # Update weights and biases
        W2 += lr * np.outer(delta2, a1)
        b2 += lr * delta2

        W1 += lr * np.outer(delta1, x)
        b1 += lr * delta1

    if epoch % 4000 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Final predictions
print("\nFinal Predictions (Full Adder):")
for x, y_true in zip(inputs, expected):
    a1 = bipolar_tanh(np.dot(W1, x) + b1)
    a2 = bipolar_tanh(np.dot(W2, a1) + b2)
    print(f"Input: {x}, SUM Pred: {a2[0]:.4f}, Exp: {y_true[0]}, "
          f"CARRY Pred: {a2[1]:.4f}, Exp: {y_true[1]}")


# Prepare predicted outputs for plotting
labels = ['000','001','010','011','100','101','110','111']
x_pos = np.arange(len(labels))

predicted = []
for x in inputs:
    a1 = bipolar_tanh(np.dot(W1, x) + b1)
    a2 = bipolar_tanh(np.dot(W2, a1) + b2)
    predicted.append(a2)

predicted = np.array(predicted)


# Plot SUM and CARRY
plt.figure(figsize=(12,6))
width = 0.35

plt.bar(x_pos - width/2, predicted[:, 0], width=width,
        label='Predicted SUM')
plt.bar(x_pos + width/2, predicted[:, 1], width=width,
        label='Predicted CARRY')

plt.xticks(x_pos, labels)
plt.ylabel('Output Value')
plt.title('Full Adder Outputs (Predicted SUM and CARRY)')
plt.legend()
plt.ylim([-1.2, 1.2])
plt.grid(True)
plt.show()
