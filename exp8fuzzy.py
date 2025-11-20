# --- Fuzzy Set Operations for Aircraft Mach Problem ---

import pandas as pd

# Given data
mach = [0.64, 0.645, 0.65, 0.655, 0.66]

# Membership values for fuzzy sets A and B
A = [0, 0.75, 1, 0.5, 0]
B = [0, 0.25, 0.75, 1, 0.5]

# Display original sets
df = pd.DataFrame({
    'Mach': mach,
    'A (near 0.65)': A,
    'B (in region 0.65)': B
})

# --- Define fuzzy operations ---
def fuzzy_union(a, b):
    return [max(x, y) for x, y in zip(a, b)]

def fuzzy_intersection(a, b):
    return [min(x, y) for x, y in zip(a, b)]

def fuzzy_complement(a):
    return [1 - x for x in a]

# --- Compute results ---
A_union_B = fuzzy_union(A, B)
A_intersect_B = fuzzy_intersection(A, B)
A_complement = fuzzy_complement(A)
B_complement = fuzzy_complement(B)
A_union_B_complement = fuzzy_union(A, B_complement)
A_intersect_B_complement = fuzzy_intersection(A, B_complement)

# --- Display results ---
results = pd.DataFrame({
    'Mach': mach,
    'A ∪ B': A_union_B,
    'A ∩ B': A_intersect_B,
    'Ā': A_complement,
    'B̄': B_complement,
    'A ∪ B̄': A_union_B_complement,
    'A ∩ B̄': A_intersect_B_complement
})

print("🔹 Original Sets:")
print(df.to_string(index=False))

print("\n🔹 Results of Fuzzy Operations:")
print(results.to_string(index=False))


# --- Fuzzy Relation & Max-Min Composition (Aircraft Example) ---

import numpy as np
import pandas as pd

# Universe of aircraft speed (X)
X = [0.72, 0.725, 0.75, 0.775, 0.78]

# Fuzzy set M = "near Mach 0.75"
M = [0, 0.8, 1, 0.8, 0]

# Universe of altitude (Y)
Y = [21, 22, 23, 24, 25, 26, 27]

# Fuzzy set N = "approximately 24,000 feet"
N = [0, 0.2, 0.7, 1, 0.7, 0.2, 0]

# -----------------------------------------------------
# (a) Construct relation R = M × N   (min operator)
# -----------------------------------------------------

R = np.zeros((len(X), len(Y)))

for i in range(len(X)):
    for j in range(len(Y)):
        R[i][j] = min(M[i], N[j])

R_df = pd.DataFrame(
    R,
    index=[f"x={x}" for x in X],
    columns=[f"y={y}" for y in Y]
)

print("🔹 Relation R = M × N (min operator):")
print(R_df)

# -----------------------------------------------------
# (b) Compute S = M1 ∘ R  (max–min composition)
# -----------------------------------------------------

# Another aircraft speed fuzzy set M1
M1 = [0, 0.8, 0.6, 0.8, 0]

S = []

for j in range(len(Y)):
    temp_vals = [min(M1[i], R[i][j]) for i in range(len(X))]
    S.append(max(temp_vals))

S_df = pd.DataFrame({
    'Altitude (k-feet)': Y,
    'μS (M1 ∘ R)': S
})

print("\n🔹 Relation S = M1 ∘ R (using max–min composition):")
print(S_df)
