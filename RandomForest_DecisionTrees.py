# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 12:20:11 2020

@author: baig
"""

########################################################### 
##### Project on Decision Trees and Random Forests  #######
###########################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('loan_data.csv')
#df.info()
#df.head()
#df.describe()

fig = plt.figure(figsize = (10, 6))
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
df[df['credit.policy']==1]['fico'].hist(alpha = 0.55, bins = 30, color = 'blue')
df[df['credit.policy']==0]['fico'].hist(alpha = 0.5, bins = 30, color = 'red')
plt.xlabel('FICO')
plt.xlim([600, 850])
plt.ylim([0, 900])
plt.legend(['Credit.policy = 1', 'Credit.policy = 0'], frameon = False)

fig = plt.figure(figsize = (10, 6))
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
df[df['not.fully.paid']==1]['fico'].hist(alpha = 0.55, bins = 30, color = 'blue')
df[df['not.fully.paid']==0]['fico'].hist(alpha = 0.5, bins = 30, color = 'red')
plt.xlabel('FICO')
plt.xlim([600, 850])
plt.ylim([0, 900])
plt.legend(['Credit.policy = 1', 'Credit.policy = 0'], frameon = False)


fig = plt.figure(figsize = (12, 8))
sns.countplot(x = df['purpose'], hue = 'not.fully.paid', data = df, palette = ['red', 'blue'])
plt.xticks(fontsize = 10)

sns.jointplot(x = 'fico', y = 'int.rate', data = df, 
              color = 'purple', size = 6, marker = 'o', s = 20)

fig = plt.figure(figsize = (5,4))
sns.lmplot(x = 'fico', y = 'int.rate', col = 'not.fully.paid', 
           hue = 'credit.policy', data = df, palette = 'Set1', scatter_kws={"s": 10})
plt.xlim([550, 850])
plt.ylim([0, 0.25])

cat_feats = ['purpose']
final_data = pd.get_dummies(df, columns = cat_feats, drop_first = True)

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 101)
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))