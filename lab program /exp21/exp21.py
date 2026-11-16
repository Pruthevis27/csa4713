
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

# Create dataset
data = np.array([
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8]
])

# Apply Factor Analysis
fa = FactorAnalysis(n_components=2)
result = fa.fit_transform(data)

# Display results
print("Original Data:")
print(data)

print("Latent Factors:")
print(result)

# Plot graph
plt.scatter(result[:, 0], result[:, 1])
plt.title("Factor Analysis")
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.show()
