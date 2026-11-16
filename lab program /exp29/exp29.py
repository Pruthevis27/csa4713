
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate random data
X = np.random.rand(100, 10)

# Create GAN Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu',
                          input_shape=(5,)),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

# Generate samples
noise = np.random.rand(5, 5)
generated = generator.predict(noise, verbose=0)

# Display generated samples
print("Generated Samples:")
print(generated)

# Plot output
plt.imshow(generated, cmap='gray')
plt.title("GAN Sample Generation")
plt.show()
