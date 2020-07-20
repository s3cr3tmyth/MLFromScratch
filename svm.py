import numpy as np
from sklearn import datasets


class SVM:

    def __init__(self, learning_rate = 0.001, lambda_p = 0.01, n_iters = 1000):

        self.lr = learning_rate
        self.lambda_p = lambda_p
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self,X,y):
        y_  = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, xi in enumerate(X):
                condition = y_[idx]*(np.dot(xi,self.w) - self.b) >= 1

                if condition:
                    self.w -= self.learning_rate* (2*self.lambda_p*self.w)

                else:
                    self.w -= self.learning_rate* (2*self.lambda_p*self.w - np.dot(xi,y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self,X):
        output = np.dot(X,self.w) - self.b
        return np.sign(output)



X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
y  = np.where(y == 0, -1, 1)


clf = SVM()
clf.fit(X,y)

print(clf.w, clf.b)