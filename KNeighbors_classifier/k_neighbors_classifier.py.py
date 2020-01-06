import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

fruits = pd.read_table('fruit_data_with_colors.txt')
# print(fruits.head(10))

# creating a mapping from fruit_label to fruit name to make it easier to identify categories

look_up_fruit = dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))
print(look_up_fruit)

# checking up the columns of the fruit table

print(fruits.columns)

# examining the data

X = fruits[['height', 'width', 'mass', 'color_score']]
Y = fruits['fruit_label']

# from matplotlib import cm
# cmap = cm.get_cmap('gnuplot')
# print(pd.__version__)
# # scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
# # scatter = pd.scatter_matrix(X_train, c= y_train)
# scatter = scatter_matrix(X, c=Y, marker = 'o', s =40, hist_kwds={'bins':20},cmap = cmap)
# plt.show()




# using KNN to classify
X = fruits[['height', 'width', 'mass']]
Y = fruits['fruit_label']

X_train, X_test , y_train , y_test = train_test_split(X,Y,random_state=0)


# # create classifer object
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=2)                       #we created here an instance object of the classifier
#
# # here we train the classifier
# knn.fit(X_train,y_train)
#
# # estimate the accuracy
# result = knn.score(X_test,y_test)
# print(result)

# further we check the sensitivity of the classifier
from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=6)
# knn.fit(X_train, y_train)
# print(knn.score(X_test, y_test))

result =[]
k = range(1,10)
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    result.append(knn.score(X_test, y_test))
    print(result)

# plotting the figure

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k, result)
plt.xticks([0,2,4,6,8,10])
plt.show()