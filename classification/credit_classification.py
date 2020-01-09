
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as nr
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

# demo to do a logistic regression\

# xseq = np.arange(-7, 7, 0.1)
# logistic = [math.exp(x)/(1 + math.exp(x)) for x in xseq]
# plt.plot(xseq, logistic, color= 'red')
# plt.show()

# now let's tango with the data
credit = pd.read_csv('data\German_Credit_Preped.csv')


# print(credit.columns)
# print(credit.shape)



# first we check for class imbalance
# y = credit['bad_credit']
# n = y.shape[0]
# count = 0
# for i in range(n):
#     if(y[i]==1):
#         count+=1
# print(count/y.shape[0])
# count = credit[['credit_history', 'bad_credit']].groupby('bad_credit').count()
# print(count)

# prepare data for scikit-learn
labels = np.array(credit['bad_credit'])

def encode_string(cat_features):
    # # first encode the string to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    # Now applying ONE-HOT-ENCODING
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1, 1))
    return encoded.transform(enc_cat_features.reshape(-1, 1)).toarray()

z = encode_string(credit['purpose'])

categorical_columns = ['credit_history', 'purpose', 'gender_status', 'time_in_residence', 'property']
Features = encode_string(credit['checking_account_status'])

for col in categorical_columns:
    temp = encode_string(credit[col])
    Features = np.concatenate([Features, temp], axis= 1)

Features = np.concatenate([Features, np.array(credit[['loan_duration_mo', 'loan_amount', 'payment_pcnt_income', 'age_yrs']])], axis=1)
# print(Features.shape)

nr.seed(42)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size= 300)
# print(indx)
X_train = Features[indx[0], :]
X_test = Features[indx[1], :]
y_train = labels[indx[0]]
y_test = labels[indx[1]]

# print(X_train[:1, ])

# Numeric features must be rescaled so that they have similar range of values
scaler = preprocessing.StandardScaler().fit(X_train[:, 34:])
X_train[:,34:] = scaler.transform(X_train[:,34:])
X_test[:,34:] = scaler.transform(X_test[:,34:])

# Fitting the logistic regression Model
logistic_model = linear_model.LogisticRegression()
logistic_model.fit(X_train, y_train)
# print(logistic_model.coef_)
probabilities = logistic_model.predict_proba(X_test)
# print(probabilities[:15, :])

# Score and evaluate classification model

































