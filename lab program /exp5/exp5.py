import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([1, 4, 9, 16, 25])

model = LinearRegression()
model.fit(X, Y)
p1 = model.predict(X)

poly = PolynomialFeatures(2)
X2 = poly.fit_transform(X)
model.fit(X2, Y)
p2 = model.predict(X2)

print("Linear Error:", mean_squared_error(Y, p1))
print("Polynomial Error:", mean_squared_error(Y, p2))

plt.scatter(X, Y)
plt.plot(X, p1, label="Linear")
plt.plot(X, p2, label="Polynomial")
plt.legend()
plt.show()
