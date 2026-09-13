import numpy as np

X = np.array([
    [-7, 6],
    [-4, 7],
    [-6, 4.5],
    [-5.5, 4],
    [-3, 4.5],
    [-6.5, 7],
    [-8, 7.5],
    [-7.5, 7],
    [-7.5, 6],
    [-4, 6],
    [-4, 7],
    [5.5, -6],
    [6.5, -6],
    [6, -6.5],
    [5.5, -6.5],
    [4.5, -6],
    [4.5, -6],
    [5.5, -3.5],
    [7, -5.5],
    [5.5, -6],
    [4, -4.5],
    [6, -4],
    [7, -2.5],
    [4.5, -9.5],
    [3, -7.5],
    [3, -5.5],
    [3.5, -4],
    [5, -2],
    [7.5, -2.5],
    [8, -3.5],
    [8, -5.5],
    [8, -7.5],
    [8, -6.5],
    [7.5, -7.5],
    [5.5, 8],
    [4.5, 6.5],
    [4, 5.5],
    [5.5, 6.5],
    [6.5, 5.5],
    [5, 8],
    [4, 6.5],
    [3.5, 6],
    [3.5, 5],
    [3.5, 6],
    [4, 4.5],
    [4.5, 4],
    [4.5, 4],
    [5, 3.5],
    [7.5, 4],
    [7.5, 6],
    [7.5, 5],
    [8.5, 5.5],
    [8.5, 6.5],
    [9, 7],
    [8.5, 7.5],
    [8, 9],
    [6.5, 6.5],
    [8.5, 8.5],
    [-9.5, -4.5],
    [-8, -3.5],
    [-6.5, -1.5],
    [-6, -3.5],
    [-5.5, -4],
    [-6, -5],
    [-6, -6],
    [-6.5, -5],
    [-7, -5.5],
    [-7.5, -6],
    [-8, -6.5],
    [-8, -7],
    [-8.5, -5.5],
    [-8.5, -5],
    [-8.5, -5],
    [-8.5, -6],
    [-8.5, -6.5],
    [-9, -6.5],
    [-9, -7],
    [-9, -8.5],
    [-8, -8],
    [-7, -8],
    [-4.5, -5],
    [-4, -4],
    [-4, -3.5],
    [-6, -7.5],
    [-5.5, -7],
    [-5, -5],
    [-4.5, -5],
    [-4.5, -4.5],
    [6, -4.5],
    [5.5, -5],
    [5.5, -4],
    [7.5, -8.5],
    [7, -8.5],
    [6.5, -8],
    [6, -7.5],
    [5, -5],
    [4.5, -4.5],
    [5.5, -3.5],
    [7.5, -3.5],
    [8.5, -2.5],
    [6.5, -2],
    [8, -2.5],
    [7, -3],
    [6, -3.5],
    [5.5, -3.5],
    [5, -4.5],
    [5, -5],
    [5.5, -5.5],
    [7, -8],
    [-7, 8.5],
    [-5.5, 9.5],
    [-7, 9.5],
    [-6, 8.5],
    [-9, 9],
    [-9, 7.5],
    [-8.5, 8],
    [-8.5, 6.5],
    [-7, 5],
    [-6, 5],
    [-5.5, 4.5],
    [-5, 6.5],
    [-4.5, 6],
    [-4.5, 7],
    [-5.5, 8.5],
    [-6, 8.5],
    [-5, 8.5],
    [-4.5, 6.5],
    [-6, 6],
    [-7, 7],
    [7.5, 7.5],
    [5.5, 7.5],
    [5, 5.5],
    [5, 4.5],
    [6.5, 4]
], dtype=float)

# 1 = same signs (QI or QIII)
# 0 = opposite signs (QII or QIV)
y = (X[:, 0] * X[:, 1] > 0).astype(int).reshape(-1, 1)

print(X.shape)
print(y.shape)


import matplotlib.pyplot as plt

# Scale inputs for stable training
X_scaled = X / 10.0
rng = np.random.default_rng(42)

# Neural network: 2 inputs -> 8 hidden neurons -> 1 output
W1 = rng.normal(0, 0.7, (2, 8))
b1 = np.zeros((1, 8))
W2 = rng.normal(0, 0.7, (8, 1))
b2 = np.zeros((1, 1))

learning_rate = 0.2
epochs = 15000
n = len(X_scaled)

for _ in range(epochs):
    # Forward pass
    hidden = np.tanh(X_scaled @ W1 + b1)
    logits = hidden @ W2 + b2
    predictions = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))

    # Backpropagation
    dz2 = predictions - y
    dW2 = hidden.T @ dz2 / n
    db2 = np.mean(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * (1 - hidden**2)
    dW1 = X_scaled.T @ dz1 / n
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # Gradient descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Training accuracy
hidden = np.tanh(X_scaled @ W1 + b1)
probabilities = 1 / (
    1 + np.exp(-np.clip(hidden @ W2 + b2, -50, 50))
)
predictions = (probabilities >= 0.5).astype(int)

accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Create a 3D prediction surface
x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 150)
x2 = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 150)
x1_grid, x2_grid = np.meshgrid(x1, x2)

grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel())) / 10.0
grid_hidden = np.tanh(grid @ W1 + b1)
grid_probability = 1 / (
    1 + np.exp(-np.clip(grid_hidden @ W2 + b2, -50, 50))
)
surface = grid_probability.reshape(x1_grid.shape)

# Plot
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    x1_grid,
    x2_grid,
    surface,
    cmap="coolwarm",
    alpha=0.7,
    edgecolor="none",
)

for label, color, name in [
    (0, "blue", "Class 0"),
    (1, "red", "Class 1"),
]:
    mask = y.ravel() == label
    ax.scatter(
        X[mask, 0],
        X[mask, 1],
        probabilities.ravel()[mask],
        color=color,
        label=name,
        s=35,
    )

ax.set_title("Neural Network Classification Surface")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Probability of same signs")
ax.set_zlim(0, 1)
ax.legend()

plt.tight_layout()
plt.show()



# ...existing code...

# Export points as a Desmos 3D list of coordinate tuples
points_3d = [
    f"({x:.8f},{y_value:.8f},{z:.8f})"
    for (x, y_value), z in zip(X, probabilities.ravel())
]

with open("desmos_points.txt", "w", encoding="utf-8") as file:
    file.write("[" + ",".join(points_3d) + "]")

# Generate the trained neural-network surface equation
terms = []

for j in range(W1.shape[1]):
    term = (
        f"({W2[j, 0]:.8f})*tanh("
        f"({W1[0, j]:.8f})*x/10+"
        f"({W1[1, j]:.8f})*y/10+"
        f"({b1[0, j]:.8f})"
        f")"
    )
    terms.append(term)

surface_equation = (
    "z=1/(1+e^(-("
    + f"{b2[0, 0]:.8f}+"
    + "+".join(terms)
    + ")))"
)

with open("desmos_surface.txt", "w", encoding="utf-8") as file:
    file.write(surface_equation)

print("Created desmos_points.txt")
print("Created desmos_surface.txt")