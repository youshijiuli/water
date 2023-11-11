import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def true_func(x):
    return 1.5 * x + 0.2


np.random.seed(10)
n_sample = 30

X_train = np.sort(np.random.rand(n_sample))
y_train = (true_func(X_train) + np.random.randn(n_sample) * 0.05).reshape(n_sample, 1)


model = LinearRegression()
model.fit(X_train[:, np.newaxis], y_train)
print(model.intercept_)
print(model.coef_)
"""[0.20879376]
[[1.49094828]]
"""
X_test = np.linspace(0, 1, 100)
plt.plot(X_test, model.predict(X_test[:, np.newaxis]), label="Model")
plt.plot(X_test, true_func(X_test), label="True function")
plt.scatter(X_train, y_train)  # 画出训练集的点
plt.legend(loc="best")
plt.show()
