
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Create dataset
X = np.random.rand(100, 3)

# Apply ICA
ica = FastICA(n_components=2, random_state=0)
result = ica.fit_transform(X)

# Display components
print("Independent Components:")
print(result[:5])

# Plot graph
plt.scatter(result[:, 0], result[:, 1])
plt.title("ICA Components")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
