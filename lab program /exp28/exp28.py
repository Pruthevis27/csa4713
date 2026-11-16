
import numpy as np
import matplotlib.pyplot as plt

# Input data
X = np.random.rand(5, 5)

# Initialize weights
W = np.random.rand(5, 5)

# Calculate hidden layer
H = 1 / (1 + np.exp(-np.dot(X, W)))

# Reconstruct data
R = 1 / (1 + np.exp(-np.dot(H, W.T)))

# Display results
print("Input Data:")
print(np.round(X, 2))

print("Reconstructed Data:")
print(np.round(R, 2))

# Display graph
plt.imshow(R, cmap='gray')
plt.title("Boltzmann Machine Reconstruction")
plt.show()
