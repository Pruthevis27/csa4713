import numpy as np
import matplotlib.pyplot as plt
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])
W1 = np.random.rand(2, 2)
W2 = np.random.rand(2, 1)
for i in range(5000):
    H = 1 / (1 + np.exp(-X @ W1))
    O = 1 / (1 + np.exp(-H @ W2))
    error = Y - O
    W2 += H.T @ (error * O * (1-O)) * 0.5
    W1 += X.T @ ((error * O * (1-O)) @ W2.T * H * (1-H)) * 0.5
print("Predicted Output:")
print(np.round(O))
plt.plot(error)
plt.show()
