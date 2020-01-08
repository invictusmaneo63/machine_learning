import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# declare the column names
columns = "age sex bmi map tc ldl hdl tch ltg glu".split()
# X,y = datasets.load_diabetes(return_X_y= True)
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=columns)
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

# print(type(X_train))
# print(X_train.shape)
# print(y.shape)
# print(df.shape[0])
# indx = range(df.shape[0])
# indx = train_test_split(indx, test_size=0.2)
# # X_train = df[indx[0]]
# print(indx[0])

# fit the model on the training data
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

# plotting the model on test data is of relevance, because now that model has been prepared it only makes sense on test data
# plt.scatter(y_test, predictions)
# plt.xlabel("True value")
# plt.ylabel("Predictions ")
# plt.show()

# let us print the model score
print("Score:", model.score(X_test, y_test))

# K-fold cross-validation
# from sklearn.model_selection import KFold
# X = np.array([[1,2],[3,4],[1,2],[3,4]])
# y = np.array([1,2,3,4])
# kf = KFold(n_splits=2)
# # kf.get_n_splits(X)
# print(kf)
# for train_index, test_index in kf.split(X):
#     print(train_index, test_index)

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
# perform 6 fold cross validation

scores = cross_val_score(model, df, y, cv=6)
print(scores)


