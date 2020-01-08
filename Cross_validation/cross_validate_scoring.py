from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

X, y = datasets.load_iris(return_X_y= True)
clf = svm.SVC(random_state=0, gamma='scale')
macro= cross_val_score(clf, X, y, cv=5, scoring= 'recall_macro')
f1 = cross_val_score(clf, X, y, cv=5, scoring= 'f1_macro')
print(f1)
print(macro)

# to check th evarious possible scoring methods applied for cross_val_score
z = sorted(sklearn.metrics.SCORERS.keys())
print(z)
