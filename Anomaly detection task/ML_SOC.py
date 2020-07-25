import numpy as np 
import numpy.random as nr
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import math
sns.set_style('whitegrid')

import os

pd.options.mode.chained_assignment = None # Warning for chained copies disabled

from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter(action='ignore')

data = pd.read_csv('annthyroid.csv')
data1 = data

data[data['binary']==0].count()/data.count()

target = data1[['binary']]
c=data1.drop(['binary'], axis=1, inplace=False)

train=c

feature=list(train.columns.values)

# f,ax=plt.subplots(figsize=(10,7))
# sns.heatmap(train.corr(), annot=True, fmt = ".2f", cmap='viridis')
# plt.title('Correlation between features', fontsize=10, weight='bold' )
# plt.show()

t=np.array(target)

x=train
y=t

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
# transform "x_train"
x_train = scaler.fit_transform(x_train)
# transforming "x_test"
x_test = scaler.transform(x_test)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.metrics import classification_report

############ Logistic Regression ################

from sklearn.linear_model import LogisticRegression

lr_c=LogisticRegression( C=1, class_weight={1:0.925, 0:0.075}, max_iter=5000,
                     penalty='l1',
                   random_state=None, solver='liblinear', verbose=0,
                   warm_start=True)
lr_c.fit(x_train,y_train.ravel())
lr_pred=lr_c.predict(x_test)
lr_ac=accuracy_score(y_test.ravel(), lr_pred)
print('Logistic Regression')
print('Accuracy: ',lr_ac)
print("AUC: ",roc_auc_score(y_test.ravel(), lr_pred))
print("F1 Score: ",f1_score(y_test.ravel(), lr_pred))
print('****************************')

########### Random Forest Classifier ###############

from sklearn.ensemble import RandomForestClassifier

rdf_c=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
rdf_c.fit(x_train,y_train.ravel())
rdf_pred=rdf_c.predict(x_test)
rdf_ac=accuracy_score(rdf_pred,y_test.ravel())

print('Random Forest Classifier')
print('Accuracy: ',rdf_ac)
print('AUC: ',roc_auc_score(y_test.ravel(), rdf_pred))
print("F1 Score: ",f1_score(y_test.ravel(), rdf_pred))

######## Plotting histogram #######

rdf_train_prob=rdf_c.predict_proba(x_train)

idx = y_train==1
idy = y_train==0

X = rdf_train_prob[idx[:,0],1]*100
Y = rdf_train_prob[idy[:,0],1]*100

bins = np.linspace(0, 100, 50)

plt.hist(X, bins, alpha=0.5, label='presence')
plt.hist(Y, bins, alpha=0.5, label='absense')
plt.yscale('log')
plt.title('Train Set')
plt.xlabel('Anomaly Score')
plt.ylabel('# Occurences')
plt.legend(loc='upper right') 
plt.show()

#########################################################

rdf_test_prob=rdf_c.predict_proba(x_test)

idx = y_test==1
idy = y_test==0

X = rdf_test_prob[idx[:,0],1]*100
Y = rdf_test_prob[idy[:,0],1]*100

bins = np.linspace(0, 100, 50)

plt.hist(X, bins, alpha=0.5, label='presence')
plt.hist(Y, bins, alpha=0.5, label='absense')
plt.yscale('log')
plt.title('Test Set')
plt.xlabel('Anomaly Score')
plt.ylabel('# Occurences')
plt.legend(loc='upper right') 
plt.show()
