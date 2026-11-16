
import numpy as np
import matplotlib.pyplot as plt

# Input data
X = np.array([[1, 0], [0, 1]])

# Initialize weights
W = np.random.rand(2, 2)

# Calculate hidden layer
H = 1 / (1 + np.exp(-np.dot(X, W)))

# Reconstruct input
R = 1 / (1 + np.exp(-np.dot(H, W.T)))

# Calculate energy
energy = -np.sum(np.dot(X, W) * H)

print("Hidden Layer:")
print(H)

print("Reconstructed Data:")
print(np.round(R))

print("Energy:", energy)

# Display graph
plt.plot(R[0], marker='o')
plt.title("RBM Reconstruction")
plt.show()
