import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import LabelEncoder

Address = input('Введите адрес квартиры: ')
Rooms = input('Количество комнат: ')
Area = input('Какова площадь квартиры? ')
Floor = input('На каком этаже находится квартира? ')
Floors = input('Сколько всего этажей в здании? ')
Type_input = input('Введите тип дома (монолитный, кирпичный, панельный) ')
Type = Type_input + ' дом'
District_input = input('В каком районе находится квартира? (Например: Медеуский, Бостандыкский) ')
District = District_input + ' р-н'
Year = input ('В каком году было построено здание? ')

Addres_list = [Address]
Rooms_list = [Rooms]
Area_list = [Area]
Floor_list = [Floor]
Floors_list = [Floors]
Type_list = [Type]
District_list = [District]
Year_list = [Year]

dict_appartments = {'Rooms': Rooms_list, 'Square_in_m^2': Area_list, 'Floor': Floor_list, 'Total Floors': Floors_list, 'Year': Year_list,"Бостандыкский р-н": 0,
                    "Ауэзовский р-н": 0, "Наурызбайский р-н": 0, "Алмалинский р-н": 0, "Алатауский р-н": 0, "Турксибский р-н": 0, "Медеуский р-н": 0, "Жетысуский р-н": 0, 
                    "кирпичный дом": 0, "монолитный дом": 0,"панельный дом": 0}

df_appartments = pd.DataFrame(dict_appartments)

x=0
for i in df_appartments.columns:
    if Type == i:
        df_appartments.loc[0,[i]]=1
    if District == i:
        df_appartments.loc[0,[i]]=1
df_appartments['Rooms'] = pd.to_numeric(df_appartments['Rooms'])
df_appartments['Square_in_m^2'] = pd.to_numeric(df_appartments['Square_in_m^2'])
df_appartments['Year'] = pd.to_numeric(df_appartments['Year'])
df_appartments['Floor'] = pd.to_numeric(df_appartments['Floor'])
df_appartments['Total Floors'] = pd.to_numeric(df_appartments['Total Floors'])

df_exp = pd.read_csv(r"C:\Users\coys7\Apartments_prices\data\final_2.csv")
df_exp.drop(df_exp[df_exp['prices'] >= 300000000].index,inplace=True)
df_exp.drop(df_exp[df_exp['Square_in_m^2'] >= 250].index,inplace=True)

all_dummies = pd.get_dummies(df_exp['District'])
all_dummies_Type = pd.get_dummies(df_exp['Type'])
df_exp_merged=df_exp.join(all_dummies)
df_exp_merged=df_exp_merged.join(all_dummies_Type)

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

y = df_exp_merged.prices

features = ['Rooms','Square_in_m^2','Floor','Total Floors', 'Year', 'Бостандыкский р-н','Ауэзовский р-н','Наурызбайский р-н','Алмалинский р-н',
            'Алатауский р-н','Турксибский р-н','Медеуский р-н','Жетысуский р-н', 'кирпичный дом', 'монолитный дом','панельный дом']
x=df_exp_merged[features]

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)

XGB = XGBRegressor(n_estimators = 1000, learning_rate=0.01)
XGB.fit(train_x, train_y, eval_set=[(val_x,val_y)], early_stopping_rounds=20)
XGB_predict = XGB.predict(df_appartments)

print("Цена квартиры: ", XGB_predict)