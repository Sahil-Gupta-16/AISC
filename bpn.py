import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ============================
# Activation Functions
# ============================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# ============================
# Sigmoid, Derivative, Softmax Plots
# ============================

x = np.linspace(-10, 10, 400)
y_sig = sigmoid(x)
y_sig_der = sigmoid_derivative(y_sig)

plt.figure(figsize=(14, 4))

# Sigmoid
plt.subplot(1, 3, 1)
plt.plot(x, y_sig, label='Sigmoid')
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.legend()

# Sigmoid Derivative
plt.subplot(1, 3, 2)
plt.plot(x, y_sig_der, label='Sigmoid Derivative', color='orange')
plt.title('Sigmoid Derivative')
plt.xlabel('x')
plt.ylabel("sigmoid'(x)")
plt.grid(True)
plt.legend()

# Softmax example
x_soft = np.array([1.0, 2.0, 3.0])
y_soft = softmax(x_soft)

plt.subplot(1, 3, 3)
plt.bar(['x1', 'x2', 'x3'], y_soft)
plt.title('Softmax Example')
plt.ylabel('softmax(x)')
plt.grid(axis='y')

plt.tight_layout()
plt.show()

# ============================
# Load Iris Dataset
# ============================

iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['species'] = iris.target_names[iris.target]

# Plot Sepal Length vs Width
plt.figure(figsize=(8, 6))
for species in iris_data['species'].unique():
    subset = iris_data[iris_data['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], label=species)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset: Sepal Length vs Sepal Width')
plt.legend()
plt.grid(True)
plt.show()

# Extract features and labels (only 2 features for simplicity)
X = iris_data.iloc[:, :2].values
y = iris_data["species"].values.reshape(-1, 1)

print("\n=== Features and Labels Shape ===")
print("X shape:", X.shape, "| y shape:", y.shape)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

print("\n=== One-hot Encoded Labels (first 5) ===")
print(y_onehot[:5])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

print("\n=== Train-Test Split ===")
print("Train samples:", len(X_train), "| Test samples:", len(X_test))

# ============================
# Initialize Weights
# ============================

np.random.seed(42)

V = np.random.uniform(-0.5, 0.5, (2, 2))  # input → hidden
v0 = np.zeros((1, 2))                     # hidden bias
W = np.random.uniform(-0.5, 0.5, (2, 3))  # hidden → output
w0 = np.zeros((1, 3))                     # output bias

print("\n=== Initial Weights and Biases ===")
print("V:\n", V)
print("v0:", v0)
print("W:\n", W)
print("w0:", w0)

# ============================
# Training
# ============================

alpha = 0.1
epochs = 20
accuracy = []

for epoch in range(1, epochs + 1):
    total_loss = 0
    print(f"\n===== Epoch {epoch} =====")

    for i in range(len(X_train)):
        x = X_train[i].reshape(1, -1)
        t = y_train[i].reshape(1, -1)

        # Forward Pass
        net_h = np.dot(x, V) + v0
        h = sigmoid(net_h)
        net_y = np.dot(h, W) + w0
        y_pred = softmax(net_y)

        # Loss (cross entropy)
        loss = -np.sum(t * np.log(y_pred + 1e-8))
        total_loss += loss

        # Backpropagation
        error_out = y_pred - t

        dW = np.dot(h.T, error_out)
        dw0 = error_out

        error_h = np.dot(error_out, W.T) * sigmoid_derivative(h)
        dV = np.dot(x.T, error_h)
        dv0 = error_h

        # Update weights
        W -= alpha * dW
        w0 -= alpha * dw0
        V -= alpha * dV
        v0 -= alpha * dv0

    # Epoch Summary
    avg_loss = total_loss / len(X_train)

    # Accuracy calculation
    correct = 0
    for j in range(len(X_train)):
        x = X_train[j].reshape(1, -1)
        t_class = y_train[j].argmax()

        net_h = np.dot(x, V) + v0
        h = sigmoid(net_h)
        net_y = np.dot(h, W) + w0
        y_pred = softmax(net_y)

        if np.argmax(y_pred) == t_class:
            correct += 1

    train_acc = correct / len(X_train)
    accuracy.append(train_acc)

    print(f"Epoch {epoch} -> Avg Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}")

# Plot epoch vs accuracy
sns.lineplot(x=range(1, epochs + 1), y=accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Epoch vs Accuracy')
plt.show()

# ============================
# Testing
# ============================

correct = 0
for i in range(len(X_test)):
    x = X_test[i].reshape(1, -1)
    t = y_test[i].argmax()

    net_h = np.dot(x, V) + v0
    h = sigmoid(net_h)
    net_y = np.dot(h, W) + w0
    y_pred = softmax(net_y)

    if np.argmax(y_pred) == t:
        correct += 1

test_accuracy = correct / len(X_test)
print(f"\n=== Final Accuracy on Test Set: {test_accuracy:.6f} ===")
