
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# CNN Model 1
model1 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(8, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# CNN Model 2
model2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compare models
for model in [model1, model2]:
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=2, verbose=0)

    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy:", accuracy)

plt.bar(["CNN 1", "CNN 2"], [
    model1.evaluate(X_test, Y_test, verbose=0)[1],
    model2.evaluate(X_test, Y_test, verbose=0)[1]
])
plt.title("CNN Comparison")
plt.ylabel("Accuracy")
plt.show()
