
import numpy as np
import matplotlib.pyplot as plt

# Input data
X = np.array([1, 2, 3, 4, 5])

# Calculate gradients
gradients = []

for i in range(len(X)):
    gradient = 2 * X[i]
    gradients.append(gradient)

# Display gradients
print("Gradients:")
print(gradients)

# Plot graph
plt.plot(gradients, marker='o')
plt.title("BPTT Gradient Analysis")
plt.xlabel("Time Steps")
plt.ylabel("Gradient")
plt.show()
