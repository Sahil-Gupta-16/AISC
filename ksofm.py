import numpy as np

def clustering_algorithm(X, vector_names, W, learning_rate, epochs):
    print("Solution:")
    print("We have 4 input vectors and we need to form 2 clusters.")
    print("So, n = 4 (input vectors) and m = 2 (clusters).\n")

    for epoch in range(epochs):
        print(f"\n================ Epoch {epoch + 1} ================\n")

        for i, x in enumerate(X):
            print(f"Input vector x = {x}\n")
            print("Calculating the Euclidean distances to the weight vectors...")

            distances = []
            for j, w in enumerate(W):
                diffs_sq = (w - x) ** 2
                dist = np.sum(diffs_sq)
                distances.append(dist)
                print(f"Distance D({j+1}): {np.round(diffs_sq, 2)}")

            print(f"\nDistance for Y1 = {distances[0]:.4f}, Distance for Y2 = {distances[1]:.4f}")

            # Determine winning neuron
            winner_index = np.argmin(distances)

            if winner_index == 0:
                print("Y1 is the closest → Y1 wins.")
                print(f"Input vector {vector_names[i]} goes to Cluster Y1.")
            else:
                print("Y2 is the closest → Y2 wins.")
                print(f"Input vector {vector_names[i]} goes to Cluster Y2.")

            print(f"Winner: Cluster Y{winner_index + 1}\n")

            # Update weight vector
            w_old = W[winner_index].copy()
            w_new = w_old + learning_rate * (x - w_old)
            W[winner_index] = w_new

            print("Updated weight matrix after this input pattern:")
            print(W)

        learning_rate *= 0.5
        print(f"\nEnd of Epoch {epoch + 1}. Updated learning rate: {learning_rate}")

    print("\n================ Final Weight Matrix ================")
    print(W)


# ---------------------------------------------------------
# 1) CASE: Different Initial Weights
# ---------------------------------------------------------

X = np.array([
    [0, 0, 1, 1],   # Vector 3
    [1, 0, 0, 0],   # Vector 8
    [0, 1, 1, 0],   # Vector 6
    [0, 0, 0, 1],   # Vector 1
    [1, 0, 1, 0]    # Vector 10
])

vector_names = ["First", "Second", "Third", "Fourth", "Fifth"]
learning_rate = 0.5

W = np.array([
    [0.2, 0.4, 0.6, 0.8],  # Y1
    [0.9, 0.7, 0.5, 0.3]   # Y2
])

print("\n\n********** CASE 1: ALL WEIGHTS DIFFERENT **********\n")
clustering_algorithm(X, vector_names, W, learning_rate, epochs=10)


# ---------------------------------------------------------
# 2) CASE: Same Initial Weights
# ---------------------------------------------------------

X = np.array([
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 1, 0]
])

vector_names = ["First", "Second", "Third", "Fourth", "Fifth"]
learning_rate = 0.5

W = np.array([
    [0.2, 0.2, 0.2, 0.2],  # Y1
    [0.9, 0.9, 0.9, 0.9]   # Y2
])

print("\n\n********** CASE 2: ALL WEIGHTS SAME **********\n")
clustering_algorithm(X, vector_names, W, learning_rate, epochs=10)
