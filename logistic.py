import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


data = datasets.load_breast_cancer()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


class Logistic():
    def __init__(self, lr = .001, itr = 1000):
        self.lr = lr
        self.itr = itr
        self.w = None 
        self.b = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) 

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.itr):

            model = np.dot(X, self.w) + self.b

            pred = self._sigmoid(model)

            # along other dimension
            dw = (1/n_samples) * np.dot(X.T, (pred - y))

            db = (1/n_samples) * np.sum(pred - y)


            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        model = np.dot(X, self.w) + self.b
        pred = self._sigmoid(model)
        pred_class = [1 if i>0.5 else 0 for i in pred]

        return pred_class


logit = Logistic(lr=0.001, itr = 1000)

logit.fit(X_train, y_train)

pred = logit.predict(X_test)

print('Accuarcy for the logistic regression is', accuracy(y_test, pred))
