import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.cross_validation import train_test_split

class KNeighborsClassifier(object):
    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _distance(self, data1, data2):
        """1: Manhattan, 2: Euclidean"""
        if self.p == 1:
            return sum(abs(data1 - data2))
        elif self.p == 2:
            return np.sqrt(sum((data1 - data2)**2))
        raise ValueError("p not recognized: should be 1 or 2")

    def _compute_weights(self, distances):
        if self.weights == 'uniform':
            return [(1, y) for d, y in distances]
        elif self.weights == 'distance':
            matches = [(1, y) for d, y in distances if d == 0]
            return matches if matches else [(1/d, y) for d, y in distances]
        raise ValueError("weights not recognized: should be 'uniform' or 'distance'")

    def _predict_one(self, test):
        distances = sorted((self._distance(x, test), y) for x, y in zip(self.X, self.y))
        weights = self._compute_weights(distances[:self.n_neighbors])
        weights_by_class = defaultdict(list)
        for d, c in weights:
            weights_by_class[c].append(d)
        return max((sum(val), key) for key, val in weights_by_class.items())[1]

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def score(self, X, y):
        return sum(1 for p, t in zip(self.predict(X), y) if p == t) / len(y)

df = pd.read_csv("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Social_Network_Ads.csv")
X = df.iloc[1:,2:4].values
y = df.iloc[1:,4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

# Fit the Naive Bayes classifier
neighbor = KNeighborsClassifier().fit(X_train, y_train)

print(neighbor.score(X_train, y_train))
print("Accuracy of KNN - ",neighbor.score(X_test, y_test)*100,"%")