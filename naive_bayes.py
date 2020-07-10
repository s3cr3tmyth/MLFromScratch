import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

X,y  = datasets.make_classification(n_samples=1000, n_features = 10, n_classes = 2, random_state = 123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc

 

class NaiveBayes:
    # no init method

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # find unqiue classes from y; y is one d array containing labels
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        ### init mean, variance and priors
        self._mean = np.zeros((n_classes,n_features), dtype=np.float64)
        self._var = np.zeros((n_classes,n_features), dtype=np.float64)

        ### each classs we want one prior
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            # samples having x class as a label
            X_c = X[c==y]

            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0) 
            # p[y] ->
            # frequency
            self._priors[c] = X_c.shape[0] / float(n_samples)
    
    def predict(self, X):
        pred = [self._predict(x) for x in X]
        return pred

    def _predict(self, x):
        posterior = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            post = prior + class_conditional
            posterior.append(post)

        return self._classes[np.argmax(posterior)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x - mean)**2/(2*var))
        denominator = np.sqrt(2*np.pi*var)

        return numerator / denominator
    

NB = NaiveBayes()
NB.fit(X_train, y_train)
pred = NB.predict(X_test)

print("accuracy is ", accuracy(y_test, pred))