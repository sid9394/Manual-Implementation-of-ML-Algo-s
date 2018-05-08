import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

class LogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])
        m = X.shape[0]

        for _ in range(self.n_iter):
            output = X.dot(self.w)
            errors = y - self._sigmoid(output)
            self.w += self.eta / m * errors.dot(X)
        return self

    def predict(self, X):
        output = np.insert(X, 0, 1, axis=1).dot(self.w)
        return [1 if i >= 0.5 else 0 for i in self._sigmoid(output)]
        #return np.where(self._sigmoid(output) >= .5, 1, 0)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)


#Enter Data
df = pd.read_csv("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Salary_Data.csv")

array = df.values
#choose tweet column
x = array[0:,np.newaxis, 1]
# print(x.shape)
y= array[0:, 0]
#print(y)

X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=.4)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=.5)
print(X_test)
reg = LogisticRegression()
reg.fit (X_train,y_train)

predicted = reg.predict(X_test)
print(predicted)

print(reg.score(X_test,y_test)*100)

#Plot
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, reg.predict(X_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()