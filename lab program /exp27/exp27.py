
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Input data
X = np.random.rand(100, 4)

# Create Deep Belief Network representation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='sigmoid',
                          input_shape=(4,)),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, X[:, :2], epochs=10, verbose=0)

# Extract features
features = model.predict(X, verbose=0)

print("Learned Representation:")
print(features[:5])

# Display graph
plt.scatter(features[:, 0], features[:, 1])
plt.title("Feature Representation")
plt.show()
