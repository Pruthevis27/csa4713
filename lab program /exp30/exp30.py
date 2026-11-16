
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Generate original and generated data
original = np.random.rand(10, 5)
generated = np.random.rand(10, 5)

# Calculate evaluation metrics
mse = mean_squared_error(original, generated)

print("Original Data:")
print(original)

print("Generated Data:")
print(generated)

print("Mean Squared Error:", mse)

# Compare outputs
plt.plot(original[0], label="Original")
plt.plot(generated[0], label="Generated")

plt.title("Generative Model Comparison")
plt.legend()
plt.show()
