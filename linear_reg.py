import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise =20, random_state = 4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# fig = plt.figure(figsize=(10,6))
# plt.scatter(X[:, 0], y, color='b',marker='o',s=30)
# plt.show()

# print(X_train[0])
# print(y_train[0])
# print(X_train.shape)
# print(X_train.shape[0])


### define perfromance measure
def MSE(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

class Regressor():

    def __init__(self, lr=0.001, iters = 1000):

        self.lr = lr
        self.iters = iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # implent gradient descent
        # init parameters
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.iters):
            # for all samples
            pred = np.dot(X, self.w) + self.b
            # one value for each feature vectoer component
            dw = (1/n_samples) * np.dot(X.T, (pred - y))

            db = (1/n_samples) * np.sum(pred - y)

            # update
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        pred = np.dot(X, self.w) + self.b
        return pred
                    

# make reg

reg = Regressor(lr = 0.05)
reg.fit(X_train, y_train)

pred = reg.predict(X_test)

mse = MSE(y_test, pred)

print('MSE', mse)