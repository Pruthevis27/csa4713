
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Create dataset
X = np.array([
    [2, 4, 6],
    [3, 5, 7],
    [4, 6, 8],
    [5, 7, 9]
])

# Apply PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)

# Reconstruct data
reconstructed = pca.inverse_transform(reduced)

print("Reduced Data:")
print(reduced)

print("Reconstructed Data:")
print(reconstructed)

# Display graph
plt.plot(X[:, 0], label="Original")
plt.plot(reconstructed[:, 0], label="Reconstructed")
plt.legend()
plt.title("PCA Reconstruction")
plt.show()
