# gaussian NB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
X, y = load_iris(return_X_y=True)
indx = range(X.shape[0])
indx = train_test_split(indx, test_size=50, random_state=42)
X_train = X[indx[0]]
X_test = X[indx[1]]
y_train = y[indx[0]]
y_test = y[indx[1]]
print(y_test)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)
n = X_test.shape[0]
print(n)
count = []
while(n>0):
    if(predictions[n-1]!=y_test[n-1]):
        count.append(1)
    else:
        count.append(0)
    n-=1
# print(count)
count = np.array(count)
print(count.sum())


