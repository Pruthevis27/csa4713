
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create dataset
x = np.linspace(0, 1, 100).reshape(-1, 1)
y = 2 * x + 1

# Optimization methods
methods = {
    "Gradient Descent": (tf.keras.optimizers.SGD, 100),
    "SGD": (tf.keras.optimizers.SGD, 1),
    "Momentum": (tf.keras.optimizers.SGD, 1),
    "Adam": (tf.keras.optimizers.Adam, 1)
}

results = {}

for name, (optimizer, batch) in methods.items():

    # Create model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(1)
    ])

    model.set_weights([
        np.array([[0.0]], dtype=np.float32),
        np.array([0.0], dtype=np.float32)
    ])

    # Select optimizer
    if name == "Momentum":
        opt = optimizer(learning_rate=0.01, momentum=0.9)
    else:
        opt = optimizer(learning_rate=0.01)

    model.compile(optimizer=opt, loss="mse")

    # Train model
    history = model.fit(
        x, y,
        epochs=100,
        batch_size=batch,
        verbose=0
    )

    results[name] = history.history["loss"]

    print(name, "Final Loss:", history.history["loss"][-1])

# Plot comparison
for name, loss in results.items():
    plt.plot(loss, label=name)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Optimizer Comparison")
plt.legend()
plt.show()
