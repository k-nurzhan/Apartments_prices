import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import LabelEncoder

df_exp = pd.read_csv(r"C:\Users\coys7\Apartments_prices\final_2.csv")

df_exp.drop(df_exp[df_exp['prices'] >= 300000000].index,inplace=True)
df_exp.drop(df_exp[df_exp['Square_in_m^2'] >= 250].index,inplace=True)

all_dummies = pd.get_dummies(df_exp['District'])
all_dummies_Type = pd.get_dummies(df_exp['Type'])
df_exp_merged=df_exp.join(all_dummies)
df_exp_merged=df_exp_merged.join(all_dummies_Type)

from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,median_absolute_error
from sklearn.model_selection import train_test_split


y = df_exp_merged.prices

features = ['Rooms','Square_in_m^2','Floor','Total Floors', 'Year', 'Бостандыкский р-н','Ауэзовский р-н','Наурызбайский р-н','Алмалинский р-н',
            'Алатауский р-н','Турксибский р-н','Медеуский р-н','Жетысуский р-н', 'кирпичный дом', 'монолитный дом','панельный дом']
x=df_exp_merged[features]

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)

def algorithm(model):
    model = model
    model.fit(train_x,train_y)
    model_prediction = model.predict(val_x)
    model_val_mae = mean_absolute_error (model_prediction, val_y)
    return model_val_mae 

RF = algorithm(RandomForestRegressor(n_estimators = 100, random_state=1))
GNB = algorithm(GaussianNB())
LR = algorithm(LogisticRegression(max_iter=200))
DTC = algorithm(tree.DecisionTreeClassifier(random_state=1))
KNN = algorithm(KNeighborsClassifier())
svc = algorithm(SVC(probability=True))

XGB = XGBRegressor(n_estimators = 1000, learning_rate=0.01)
XGB.fit(train_x, train_y, eval_set=[(val_x,val_y)], early_stopping_rounds=20)
XGB_predict = XGB.predict(val_x)
XGB_val_mae = mean_absolute_error(XGB_predict, val_y)

print("MAE of Random Forest Regressor: {:.3f}".format(RF))
print("MAE of XGBoost Regressor: {:.3f}".format(XGB_val_mae))
print("MAE of Gaussian Naive Bayes: {:.3f}".format(GNB))
print("MAE of Logistic Regression: {:.3f}".format(LR))
print("MAE of Decision Tree Classifier: {:.3f}".format(DTC))
print("MAE of K-nearest Neighbors: {:.3f}".format(KNN))
print("MAE of Support Vector Classification: {:.3f}".format(svc))

models = ["Random Forest Regressor", "XGBoost", "Gaussian Naive Bayes", "Logistic Regression", "Decision Tree Regressor", "K-nearest Neighbors", "Support Vector"]
tests_mae =[RF, XGB_val_mae, GNB, LR, DTC, KNN, svc]
compare_models = pd.DataFrame({"Algorithms": models, "MAE": tests_mae})
compare_models.sort_values(by = "MAE", ascending=True)

print("Best algorithm: \n",compare_models[compare_models['MAE']==compare_models['MAE'].min()])