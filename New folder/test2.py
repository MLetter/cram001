
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

# reading the data
df = pd.read_csv("cleanCredit.csv", delimiter=",")
df = df.apply (pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)


#print(datainput)
data=df



#Assigns X and y for training 

X = data.drop(['Credit_Utilization_Ratio', 'Total_EMI_per_month','Num','Credit_Score'], axis='columns')
y1 = data['Credit_Score']



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


'''
log_scores = cross_val_score(LogisticRegression(max_iter=1000000), X_scaled, y1)
sgd_scores = cross_val_score(SGDClassifier(), X_scaled, y1)
svc_scores = cross_val_score(SVC(), X_scaled, y1)
rf_scores = cross_val_score(RandomForestClassifier(), X_scaled, y1)

print('Log score: ' + str(log_scores.mean()))
print('SGD score: ' + str(sgd_scores.mean()))
print('SVC score: ' + str(svc_scores.mean()))
print('RF score: ' + str(rf_scores.mean()))
'''



#this modual will look fot the best paramiters for the model
#rf_description = pd.DataFrame(rf_model.cv_results_)
#print(rf_description[['param_criterion', 'param_max_features', 'param_n_estimators', 'mean_test_score']])
#print(rf_model.best_params_)
# PRINT () << C!! IN
#PRINT rfr best_model = RandomforestRegressor(n_estemators=100)
#trains the model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y1, test_size=0.2, random_state=0)


#input the best params 
rfr_best_model = RandomForestRegressor(n_estimators=100, max_features='sqrt', criterion='squared_error')
#rfr_best_model = LogisticRegression()
rfr_best_model.fit(X_train, y_train)
print(rfr_best_model.fit(X_train, y_train))
y_pred = rfr_best_model.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)


'''
#plotting the predicted model 
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
'''



testdata = [[27, 50000, 2500, 1, 2, 4, 4,3, 50000, 0, 500]]
#testdata = [[30, 34081, 2611, 8, 7, 15, 3,9, 1704, 29, 411]]
#Make prediction
predicted_risk_level = rfr_best_model.predict(testdata)
print(predicted_risk_level)
if predicted_risk_level > 2.49:
    print("Good")
elif 2.5 > predicted_risk_level > 1.49:
    print("Standard")
else:
    print("bad")







