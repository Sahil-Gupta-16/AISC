import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# ======================================================
# DATA PREPROCESSING
# ======================================================

df = pd.read_csv("/content/Bengaluru_House_Data.csv")
df = df.dropna()

# ---- Convert size like "3 BHK" → 3 ----
df['rooms'] = df['size'].apply(lambda x: int(x.split()[0]))

# ---- Clean total_sqft ----
def convert_sqft(x):
    try:
        return float(x)
    except:
        if "-" in x:
            a, b = x.split("-")
            return (float(a) + float(b)) / 2
        return None

df["size_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna(subset=["size_sqft"])

# ---- Encode categorical columns ----
label_cols = ["area_type", "availability", "location", "society"]
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# ---- Feature selection ----
features = [
    "area_type",
    "availability",
    "location",
    "rooms",
    "size_sqft",
    "bath",
    "balcony"
]

input_data = df[features].values
output_data = df["price"].values.reshape(-1, 1)

# ---- Train/Test ----
X_train, X_test, y_train, y_test = train_test_split(
    input_data, output_data, test_size=0.2, random_state=42
)

# ---- Normalize ----
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)


# ======================================================
# ANFIS MODEL
# ======================================================

class ANFIS:
    def __init__(self, X, y, n_rules=8):
        self.X = X
        self.y = y
        self.n_rules = n_rules
        self.n_inputs = X.shape[1]

        # Gaussian MF: (center, sigma)
        self.mf = np.random.rand(self.n_inputs, n_rules, 2)

        # Consequent parameters: a1..a7 and b
        self.cons = np.random.randn(n_rules, self.n_inputs + 1)

    def gaussian(self, x, c, s):
        return np.exp(-((x - c) ** 2) / (2 * s ** 2))

    def forward(self, X):
        N = X.shape[0]
        mu = np.zeros((N, self.n_rules))

        # Rule firing strengths
        for r in range(self.n_rules):
            prod = 1
            for i in range(self.n_inputs):
                c = self.mf[i, r, 0]
                s = max(self.mf[i, r, 1], 0.01)
                prod *= self.gaussian(X[:, i], c, s)
            mu[:, r] = prod

        # Normalize
        w = mu / (np.sum(mu, axis=1, keepdims=True) + 1e-9)

        # Consequent z = a1*x1 + ... + a7*x7 + b
        Z = np.zeros((N, self.n_rules))
        for r in range(self.n_rules):
            a = self.cons[r, :-1]
            b = self.cons[r, -1]
            Z[:, r] = X @ a + b

        # Final output
        y_pred = np.sum(w * Z, axis=1, keepdims=True)
        return y_pred, w, Z

    def train(self, epochs=1000, lr=0.0005):
        for epoch in range(epochs):
            y_pred, w, Z = self.forward(self.X)
            error = self.y - y_pred

            # =============================
            # UPDATE MEMBERSHIP FUNCTIONS
            # =============================
            for i in range(self.n_inputs):
                for r in range(self.n_rules):
                    c = self.mf[i, r, 0]
                    s = max(self.mf[i, r, 1], 0.01)
                    x = self.X[:, i]
                    g = self.gaussian(x, c, s)

                    dcd = g * (x - c) / (s**2)
                    dsd = g * ((x - c) * 2) / (s * 3)

                    grad_c = np.sum(error.flatten() * dcd)
                    grad_s = np.sum(error.flatten() * dsd)

                    grad_c = np.clip(grad_c, -10, 10)
                    grad_s = np.clip(grad_s, -10, 10)

                    self.mf[i, r, 0] += lr * grad_c
                    self.mf[i, r, 1] += lr * grad_s
                    self.mf[i, r, 1] = max(self.mf[i, r, 1], 0.01)

            # =============================
            # UPDATE CONSEQUENT PARAMS
            # =============================
            for r in range(self.n_rules):
                weighted_error = (error.flatten() * w[:, r]).reshape(-1, 1)
                grad_a = -np.sum(weighted_error * self.X, axis=0)
                grad_b = -np.sum(error.flatten() * w[:, r])

                grad_a = np.clip(grad_a, -10, 10)
                grad_b = np.clip(grad_b, -10, 10)

                self.cons[r, :-1] -= lr * grad_a
                self.cons[r, -1] -= lr * grad_b

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Error = {np.mean(np.abs(error))}")

    def predict(self, X):
        y_pred, _, _ = self.forward(X)
        return y_pred


# ======================================================
# TRAIN
# ======================================================

anfis = ANFIS(X_train, y_train, n_rules=8)
anfis.train(epochs=1000, lr=0.0005)

# Predict
y_pred = anfis.predict(X_test)
y_pred = np.clip(y_pred, 0, None)

# De-normalize
y_pred_real = scaler_y.inverse_transform(y_pred)
y_test_real = scaler_y.inverse_transform(y_test)

# ======================================================
# PLOT RESULTS
# ======================================================

plt.scatter(y_test_real, y_pred_real, alpha=0.5)
plt.plot([min(y_test_real), max(y_test_real)],
         [min(y_test_real), max(y_test_real)], color='red')

plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("ANFIS Regression - All Features")
plt.show()
