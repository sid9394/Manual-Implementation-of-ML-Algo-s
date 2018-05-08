import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

class LinearRegressionSGD(object):
    def __init__(self, eta=0.1, n_iter=50, shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.w = np.ones(X.shape[1])

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            for x, target in zip(X, y):
                output = x.dot(self.w)
                error = target - output
                self.w += self.eta * error * x
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w)

    def score(self, X, y):
        return 1 - sum((self.predict(X) - y)**2) / sum((y - np.mean(y))**2)

#Enter Data
df = pd.read_csv("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Salary_Data.csv")

array = df.values
#choose tweet column
x = array[0:,np.newaxis, 1]
# print(x.shape)
y= array[0:, 0]
#print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

reg = LinearRegressionSGD(eta=.1, n_iter=1500)
reg.fit (X_train,y_train)

# results = model_selection.cross_val_score(reg, x, y)
# print(results.mean())
predicted = reg.predict(X_test)
print(predicted)

print(reg.score(X_test,y_test)*100)

#Plot
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, reg.predict(X_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
