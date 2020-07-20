import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


# generate data

np.random.seed(20)
num_obs = 5000

x1 = np.random.multivariate_normal([0,0],[[1,.75],[.75,1]],num_obs)
x2 = np.random.multivariate_normal([1,4],[[1,.75],[.75,1]],num_obs)

# len(x1)

features = np.vstack((x1,x2)).astype(np.float32)
labels = np.hstack((np.zeros(num_obs),np.ones(num_obs)))

# len(labels)
plt.figure(figsize=(12,8))
plt.scatter(features[:,0], features[:,1], c=labels,alpha=0.4)


# def sigmoid; converting scores into probabilities 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# log liklihood fucntion
def log_liklihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target*scores - np.log(1 + np.exp(scores)))
    return ll


def log_R(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
       intercept = np.ones((features.shape[0],1)) 
       features = np.hstack((intercept,features))

    weights = np.zeros(features.shape[1])

    for i in range(num_steps):
        scores = np.dot(features, weights)
        probs = sigmoid(scores)

        # update weighyts with gradients
        output = target - probs
        gradient = np.dot(features.T, output)
        weights += learning_rate * gradient

        if num_steps % 10000 == 0:
            print(log_liklihood(features, target, weights))

    return weights

weights = log_R(features, labels,
                     num_steps = 300000, learning_rate = 5e-5, add_intercept=True)