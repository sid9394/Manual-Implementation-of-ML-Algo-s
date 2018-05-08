import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

class GaussianNB(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)]
                    for i in separated])
        return self

    def _prob(self, x, mean, std):
        exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
        return np.log(exponent / (np.sqrt(2 * np.pi) * std))

    def predict_log_proba(self, X):
        return [[sum(self._prob(i, *s) for s, i in zip(summaries, x))
                for summaries in self.model] for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)


df = pd.read_csv("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Social_Network_Ads.csv")
X = df.iloc[1:,2:4].values
y = df.iloc[1:,4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

# Fit the Naive Bayes classifier
nb = GaussianNB().fit(X_train, y_train)
print(nb.score(X_train, y_train))
print("Accuracy of Naive Bayes - ",nb.score(X_test, y_test)*100,"%")
