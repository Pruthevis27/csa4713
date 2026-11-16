
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Dataset
X = np.array([
    [2, 4, 6],
    [3, 5, 7],
    [4, 6, 8],
    [5, 7, 9]
])

# Apply PCA
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# Display results
print("PCA Result:")
print(result)

print("Explained Variance:")
print(pca.explained_variance_ratio_)

# Plot graph
plt.plot(pca.explained_variance_ratio_, marker='o')
plt.title("Explained Variance")
plt.xlabel("Components")
plt.ylabel("Variance")
plt.show()
