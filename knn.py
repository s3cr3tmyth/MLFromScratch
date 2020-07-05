import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from collections import Counter
### import iris datasets

iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1234)

# print(X_train.shape[0])

## write dist fnct
def euDist(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

### write KNN function

class KNN:

    def __init__(self, k):
        self.k = k
    ## heree fit doesn't involve any training step 
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    ### we will use predict helper function to process single sample
    def _predict(self, x):
        # compute distances
        dist = [euDist(x,x_train) for x_train in self.X_train]
        # get the k nearest samples, labels
        top_k_samples = np.argsort(dist)[:self.k]   
        top_k_labels = [self.y_train[i] for i in top_k_samples]    
        # majority vote
        # here most common returns list of tuple of value and count, so for value we use [0][0]
        most_common = Counter(top_k_labels).most_common(1)

        return most_common[0][0]


### make a classifier

clf = KNN(k=5)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = np.sum(pred == y_test) / len(y_test)

print("Accuracy", acc)