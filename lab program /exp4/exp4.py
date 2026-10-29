import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

P = np.array([0.5, 0.5])
Q = np.array([0.8, 0.2])

print("Entropy:", entropy(P, base=2))
print("Cross Entropy:", -np.sum(P * np.log2(Q)))
print("KL Divergence:", entropy(P, Q, base=2))

plt.bar(['P1', 'P2'], P, label='P')
plt.bar(['Q1', 'Q2'], Q, alpha=0.5, label='Q')
plt.legend()
plt.show()
