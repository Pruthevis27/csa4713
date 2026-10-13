
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create a dataset
data = {
    'X': [2, 4, 6, 8, 10],
    'Y': [1, 3, 5, 7, 9]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

# Step 2: Calculate covariance matrix
cov_matrix = df.cov()
print("\nCovariance Matrix:")
print(cov_matrix)

# Step 3: Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Step 4: Visualize the dataset
plt.scatter(df['X'], df['Y'])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Covariance and Data Analysis")
plt.grid(True)
plt.show()
