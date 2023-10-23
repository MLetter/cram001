# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:56:29 2023

@author: mstte
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, auc, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

data = pd.read_csv("cleanCredit.csv")


data = data.fillna(0)
numerical_type = [
    'Age', 'Annual_Income', 'Num_of_Loan',
    'Outstanding_Debt','Amount_invested_monthly', 'Monthly_Balance'
]
for feature in numerical_type:
    data[feature] = data[feature].replace('[^-0-9.]', '', regex=True).astype("float64", errors='ignore')
data['Age'] = data['Age'][(10 < data['Age']) & (data['Age'] < 100)]
    

    
one_hot = ["Credit_Mix"]
one_hot_features = pd.get_dummies(data[one_hot])
data = data.drop(columns=one_hot)


    
y = data['Credit_Score']
x = data.drop(columns=["Credit_Score"])
X = x.fillna(x.median())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=634356)

X_train.dtypes

X_train = X_train.join(one_hot_features)
X_test = X_test.join(one_hot_features)

y_train = y_train[X_train.index]


clf = RandomForestClassifier(max_depth=3, random_state=4536574, n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)
print(f"Train score: {clf.score(X_train, y_train)}, test score: {clf.score(X_test, y_test)}")


print("Recall for labels: ", recall_score(pd.Series(clf.predict(X_test)), y_test, average=None))
print("Precision for labels: ", precision_score(pd.Series(clf.predict(X_test)), y_test, average=None))
print("F1 for labels: ", f1_score(pd.Series(clf.predict(X_test)), y_test, average=None))



rf_modelx = GridSearchCV(RandomForestClassifier(), {
    'n_estimators': [25, 50, 75, 100, 150, 200],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
})

rf_model = rf_modelx.fit(X, y)

rfr_best_model = RandomForestRegressor(n_estimators=100, max_features='sqrt', criterion='squared_error')
rfr_best_model.fit(X_train, y_train)

y_pred = rfr_best_model.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)





