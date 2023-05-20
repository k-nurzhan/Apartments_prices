import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import LabelEncoder

df_final = pd.read_csv(r"C:\Users\coys7\Desktop\krisha\final_2")
df_exp = pd.read_csv(r"C:\Users\coys7\Desktop\krisha\final_21")

df_exp = df_exp.dropna(axis=0)


df_exp['Year']=df_exp['Year'].str.replace('г.п.', '', regex=True)
df_exp['Year']=df_exp['Year'].str.replace('[', '', regex=True)
df_exp['Year']=df_exp['Year'].str.replace(']', '', regex=True)
df_exp['Year']=df_exp['Year'].str.replace("'", '', regex=True)
df_exp['Type']=df_exp['Type'].str.replace('[', '', regex=True)
df_exp['Type']=df_exp['Type'].str.replace(']', '', regex=True)
df_exp['Type']=df_exp['Type'].str.replace("'", '', regex=True)
df_exp['District']=df_exp['District'].str.replace("'", '', regex=True)
df_exp['District']=df_exp['District'].str.replace('[', '', regex=True)
df_exp['District']=df_exp['District'].str.replace(']', '', regex=True)
df_exp['prices']=df_exp['prices'].str.replace('[\xa0]', '', regex=True)
df_exp['prices']=df_exp['prices'].str.replace('〒', '', regex=True)
df_exp['Floor']=df_exp['Floor'].str.replace(r'/[0-9]+[0-9]', '', regex=True)
df_exp['Floor']=df_exp['Floor'].str.replace(r'/[0-9]', '', regex=True)
df_exp = df_exp.dropna(axis=0)

filter=df_exp['Type'].isin(["монолитный дом", "панельный дом", "кирпичный дом"])

df_exp[filter]

df_exp=df_exp[filter]

df_exp = df_exp.drop_duplicates()

filter_2=df_exp['District'].isin(["Бостандыкский р-н", "Ауэзовский р-н", "Наурызбайский р-н", "Алмалинский р-н", "Алатауский р-н", "Турксибский р-н", "Медеуский р-н", "Жетысуский р-н"])

df_exp=df_exp[filter_2]

#categorizing Type and District into numbers
le= LabelEncoder()

Type_label=le.fit_transform(df_exp['Type'])
df_exp['Type_code'] = Type_label
df_exp['Type_code'].value_counts()

District_label =le.fit_transform(df_exp['District'])
df_exp['District_code'] = District_label
df_exp['District_code'].value_counts()

from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

y = df_exp.prices

features = ['Rooms', 'Square_in_m^2', 'Floor', 'Type_code', 'Year', 'District_code']
x=df_exp[features]

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)


rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(train_x, train_y)
rf_prediction = rf_model.predict(val_x)

rf_val_mae = mean_absolute_error (rf_prediction, val_y)
print(rf_val_mae)

gnb = GaussianNB()
gnb.fit(train_x,train_y)
gnb_prediction = gnb.predict(val_x)
gnb_prediction_float = np.array(gnb_prediction, dtype=float)

gnb_val_mae = mean_absolute_error(gnb_prediction_float, val_y)
print(gnb_val_mae)

lr =LogisticRegression(max_iter = 2000)
lr.fit(train_x, train_y)
lr_prediction = lr.predict(val_x)
lr_prediction_float = np.array(lr_prediction, dtype=float)

lr_val_mae = mean_absolute_error(lr_prediction_float, val_y)
print(lr_val_mae)

dt = tree.DecisionTreeClassifier(random_state=0)
dt.fit(train_x,train_y)
dt_prediction = dt.predict(val_x)

dt_val_mae = mean_absolute_error(dt_prediction,val_y)
print(dt_val_mae)

knn = KNeighborsClassifier()
knn.fit(train_x,train_y)
knn_prediction = knn.predict(val_x)

knn_val_mae = mean_absolute_error(knn_prediction, val_y)
print(knn_val_mae)

svc = SVC (probability=True)
svc.fit(train_x,train_y)
svc_prediction = svc.predict(val_x)
svc_prediction_float = np.array(svc_prediction, dtype=float)

svc_val_mae = mean_absolute_error(svc_prediction,val_y)
print(svc_val_mae)