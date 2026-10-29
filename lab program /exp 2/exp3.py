import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

x = np.arange(0, 6)
y = binom.pmf(x, 5, 0.5)

print("Probabilities:", y)

# Bayes' Theorem
PA = 0.4
PBA = 0.8
PB = 0.5

print("Bayes Probability:", (PBA * PA) / PB)

plt.bar(x, y)
plt.title("Binomial Distribution")
plt.show()
