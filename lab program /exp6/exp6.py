import numpy as np
import matplotlib.pyplot as plt

x = 5.0
lr = 0.1
loss = []

for i in range(20):
    x = x - lr * 2 * x
    loss.append(x**2)

print("Gradient Descent Result:", x)

plt.plot(loss)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Gradient Descent")
plt.show()
