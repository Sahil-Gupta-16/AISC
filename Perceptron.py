import numpy as np

class Perceptron:

    def __init__(self, num_inputs, learning_rate=0.01):
        # +1 for bias weight
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = learning_rate

    # Linear layer
    def linear(self, inputs):
        return inputs @ self.weights[1:] + self.weights[0]

    # Heaviside step function
    def Heaviside_step_fn(self, z):
        return 1 if z >= 0 else 0

    # Prediction
    def predict(self, inputs):
        Z = self.linear(inputs)

        # If Z is a vector (batch)
        if isinstance(Z, np.ndarray):
            return [self.Heaviside_step_fn(z) for z in Z]

        # If Z is scalar
        return self.Heaviside_step_fn(Z)

    # Loss = prediction - target
    def loss(self, prediction, target):
        return prediction - target

    # Single training step
    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = self.loss(prediction, target)

        print(f"Prediction: {prediction}")
        print(f"Error: {error}")
        print(f"Initial weights: {self.weights[1:]}")
        print(f"Initial bias: {self.weights[0]}")

        # Update rule
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error

        print(f"Updated weights: {self.weights[1:]}")
        print(f"Updated bias: {self.weights[0]}")
        print()

    # Fit model for multiple epochs
    def fit(self, X, y, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            for inputs, target in zip(X, y):
                print(f"Inputs: {inputs}")
                print(f"Target: {target}")
                self.train(inputs, target)
            print()



#cell2

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate linearly separable dataset
X, y = make_blobs(
    n_samples=5,
    n_features=2,
    centers=2,
    cluster_std=3,
    random_state=23
)

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23, shuffle=True
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nX_train:\n", X_train)
print("\ny_train:\n", y_train)
print("\nX_test:\n", X_test)
print("\ny_test:\n", y_test)

# Fix random seed
np.random.seed(23)

# Initialize perceptron
perceptron = Perceptron(num_inputs=X_train.shape[1])

# Train for 2 epochs
perceptron.fit(X_train, y_train, num_epochs=2)

# Predict on test set
pred = perceptron.predict(X_test)

# Correct accuracy formula (pred == y_test)
accuracy = np.mean(pred == y_test)

print("\nAccuracy:", accuracy)

# Plot test data with predicted colors
plt.scatter(X_test[:, 0], X_test[:, 1], c=pred)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title("Perceptron Predictions on Test Data")
plt.show()
