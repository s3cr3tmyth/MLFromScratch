import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

data = datasets.load_iris()
X = data.data 
y = data.target

class PCA:

    def __init__(self, n_componenets):
        self.n_c = n_componenets
        self.eigen = None
        self.mean = None


    def fit(self, X):

        # standardize the data
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean
        # find covariance matrix
        cov = np.cov(X.T)
        # find eigenvalues and vectors 
        eigenval, eigenvec = np.linalg.eig(cov)
        # eigen vectors are returned as a column vector
        eigenvec = eigenvec.T
        # sorting in descnding order
        ids = np.argsort(eigenval)[::-1]
        eigenvec = eigenvec[ids]
        eigenval = eigenval[ids]
        # find top n eigenvalues
        self.eigen = eigenvec[0:self.n_c]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.eigen.T)



pca = PCA(5)
pca.fit(X)
projected = pca.transform(X)

comp1 = projected[:, 0]
comp2 = projected[:, 1]


plt.scatter(comp1, comp2, c = y, edgecolor = 'none', alpha = 0.9, cmap = plt.cm.get_cmap('viridis',3))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()
