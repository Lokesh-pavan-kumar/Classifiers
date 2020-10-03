# # Importing the required packages / libraries
#
# * Numpy is used to perform calculations on the datasets to
#   prepare the classifier.

import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error


class LinearRegression:

    def __init__(self):
        self.X = None  # The feature vectors [shape = (m, n) => (n, m)]
        self.y = None  # The regression outputs [shape = (m, 1)]
        self.W = None  # The parameter vector `W` [shape = (n, 1)]
        self.bias = None
        self.lr = None  # Learning Rate `alpha`
        self.m = None
        self.n = None
        self.epochs = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1500, use_bias: bool = True, lr: float = 0.05):
        (self.m, self.n) = X.shape
        self.X = (X - X.mean(axis=0)) / X.std(axis=0)

        assert y.shape == (self.m, 1) or (self.m,)
        self.y = np.expand_dims(y, axis=1)
        self.W = np.zeros((self.n, 1))
        if use_bias:
            self.bias = 0.0
        self.epochs = epochs
        self.lr = lr
        return self.minimize()

    def minimize(self):
        cost_hist = np.zeros((self.epochs,))
        for num_epoch in range(self.epochs):
            if self.bias is None:
                predictions = np.dot(self.X, self.W)
            else:
                predictions = np.dot(self.X, self.W) + self.bias

            grad_w = np.dot(self.X.T, (predictions - self.y))
            self.W = self.W - (self.lr / self.m) * grad_w
            if self.bias is not None:
                grad_b = np.sum(predictions - self.y)
                self.bias = self.bias - (self.lr / self.m) * grad_b

            loss = (1 / (2 * self.m)) * np.sum(np.square(predictions - self.y))
            cost_hist[num_epoch] = loss
            print(f'Epoch : {num_epoch+1}/{self.epochs} \t Loss : {loss}')
        return cost_hist

    def predict(self, x: np.ndarray):
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        if self.bias is None:
            return np.dot(x, self.W)
        else:
            return np.dot(x, self.W) + self.bias


def compute_cost(x, y, theta):
    x = np.hstack([np.ones((x.shape[0], 1)), x])
    predictions = np.dot(x, theta)
    return 1 / (2 * x.shape[0]) * np.sum(np.square(predictions - y))


linear_regression = LinearRegression()
X_train, y_train = load_boston(True)
linear_regression.fit(X_train, y_train, 1500, lr=0.01)
train_predictions = linear_regression.predict(X_train)

error = np.sum(np.square(train_predictions - y_train.reshape((-1, 1)))) / (2*y_train.size)
print(error)
print(mean_squared_error(train_predictions, y_train) / 2)