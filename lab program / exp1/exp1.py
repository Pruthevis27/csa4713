import numpy as np
import matplotlib.pyplot as plt
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
print("Vector Addition:", v1 + v2)
print("Dot Product:", np.dot(v1, v2))
print("Matrix Addition:\n", A + B)
print("Matrix Multiplication:\n", np.dot(A, B))
print("Transpose of A:\n", A.T)
plt.imshow(A, cmap="viridis")
plt.colorbar()
plt.title("Matrix A")
plt.show()
