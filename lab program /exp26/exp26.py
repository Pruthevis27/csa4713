
import numpy as np
import matplotlib.pyplot as plt

# Initialize data and weights
v = np.array([1, 0])
W = np.random.rand(2, 2)

# Gibbs Sampling
h = 1 / (1 + np.exp(-np.dot(v, W)))
v_new = 1 / (1 + np.exp(-np.dot(h, W.T)))

# Contrastive Divergence
error = v - v_new

print("Hidden Layer:", h)
print("Reconstructed Data:", np.round(v_new))
print("Weight Update:", error)

# Plot graph
plt.bar(["Input 1", "Input 2"], v)
plt.title("Gibbs Sampling")
plt.show()
