import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
np.random.seed(42)
# X = np.hstack((X, 2* np.random.random((X.shape[0], 36))))
# print(X.shape)

clf = Pipeline([('anova', SelectPercentile(chi2)),
                ('scaler', StandardScaler()),
                ('svc', SVC(gamma="auto"))])

# plot the cross validation score as a function of percentile of features
score_means = []
score_stds = []
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 50, 60, 80, 100)
# print(type(percentiles))

for percentile in percentiles:
    clf.set_params(anova__percentile = percentile)
    score = cross_val_score(clf, X, y, cv = 3)
    score_means.append(score.mean())
    score_stds.append(score.std())

# plt.errorbar(percentiles, score_means, np.array(score_stds))
# plt.xticks(np.linspace(0, 100, 11, endpoint=True))
# plt.show()

plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title(
    'Performance of the SVM-Anova varying the percentile of features selected')
plt.xticks(np.linspace(0, 100, 11, endpoint=True))
plt.xlabel('Percentile')
plt.ylabel('Accuracy Score')
plt.axis('tight')
plt.show()








# m = chi2(X,y)
# q = np.array(m[0])
# r = np.array(m[1])
# print(r,q)