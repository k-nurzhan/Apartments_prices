import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df_final = pd.read_csv(r"C:\Users\coys7\Desktop\krisha\final_21")

df_final = df_final.dropna(axis=0)

df_final['Year']=df_final['Year'].str.replace('г.п.', '', regex=True)
df_final['Year']=df_final['Year'].str.replace('[', '', regex=True)
df_final['Year']=df_final['Year'].str.replace(']', '', regex=True)
df_final['Year']=df_final['Year'].str.replace("'", '', regex=True)
df_final['Type']=df_final['Type'].str.replace('[', '', regex=True)
df_final['Type']=df_final['Type'].str.replace(']', '', regex=True)
df_final['Type']=df_final['Type'].str.replace("'", '', regex=True)
df_final['District']=df_final['District'].str.replace("'", '', regex=True)
df_final['District']=df_final['District'].str.replace('[', '', regex=True)
df_final['District']=df_final['District'].str.replace(']', '', regex=True)
df_final['prices']=df_final['prices'].str.replace('[\xa0]', '', regex=True)
df_final['prices']=df_final['prices'].str.replace('〒', '', regex=True)
df_final['Floor']=df_final['Floor'].str.replace(r'/[0-9]+[0-9]', '', regex=True)
df_final['Floor']=df_final['Floor'].str.replace(r'/[0-9]', '', regex=True)
df_final = df_final.dropna(axis=0)

filter=df_final['Type'].isin(["монолитный дом", "панельный дом", "кирпичный дом"])

df_final[filter]

df_final=df_final[filter]

df_final = df_final.drop_duplicates()

filter_2=df_final['District'].isin(["Бостандыкский р-н", "Ауэзовский р-н", "Наурызбайский р-н", "Алмалинский р-н", "Алатауский р-н", "Турксибский р-н", "Медеуский р-н", "Жетысуский р-н"])

df_final=df_final[filter_2]

#categorizing Type and District into numbers
le= LabelEncoder()

Type_label=le.fit_transform(df_final['Type'])
df_final['Type_code'] = Type_label
df_final['Type_code'].value_counts()

District_label =le.fit_transform(df_final['District'])
df_final['District_code'] = District_label
df_final['District_code'].value_counts()

df_final['Floor'] = pd.to_numeric(df_final['Floor'])
df_final['Year'] = pd.to_numeric(df_final['Year'])
df_final['prices'] = pd.to_numeric(df_final['prices'])

df_final.to_csv('final_22', index=False)