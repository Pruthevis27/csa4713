
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Input data
X = np.array([[[1], [2], [3]],
              [[2], [3], [4]]], dtype=float)

Y = np.array([4, 5], dtype=float)

# Create RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3, 1)),
    tf.keras.layers.SimpleRNN(5),
    tf.keras.layers.Dense(1)
])

# Compile with gradient clipping
model.compile(
    optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
    loss='mse'
)

# Train model
history = model.fit(X, Y, epochs=20, verbose=0)

print("Training Completed")
print("Final Loss:", history.history['loss'][-1])

# Plot loss
plt.plot(history.history['loss'])
plt.title("RNN Gradient Clipping")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
