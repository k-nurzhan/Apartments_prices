import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder

df_first_page = pd.read_csv(r"C:\Users\coys7\Apartments_prices\app-s_first_page")
df_other_pages = pd.read_csv(r"C:\Users\coys7\Apartments_prices\app-s_other_pages")
df_merged = pd.concat([df_other_pages,df_first_page])


#splitiing main info into different columns
df_merged[['Rooms', 'Square_in_m^2', 'floor']]=df_merged['main_info'].str.split(",", expand=True)


#cleaning data from unnecessary info
df_merged['prices']=df_merged['prices'].str.replace('[\n ]', '', regex=True)
df_merged['address']=df_merged['address'].str.replace('[\n]', '', regex=True)
df_merged['description']=df_merged['description'].str.replace('[\n]', '', regex=True)
df_merged['Rooms']=df_merged['Rooms'].str.replace('-комнатная квартира', '', regex=True)
df_merged['Square_in_m^2']=df_merged['Square_in_m^2'].str.replace(' м²', '', regex=True)
df_merged['floor']=df_merged['floor'].str.replace(' этаж', '', regex=True)
df_merged['prices']=df_merged['prices'].str.replace('от', '', regex=True)
df_merged['address']=df_merged['address'].str.replace('   ', '', regex=True)
df_merged['address']=df_merged['address'].str.replace('["]', '', regex=True)


#working with description, finding pattern, to extract necessary information
pattern_1 = r'[а-яА-ЯёЁ]+ дом'              #building type
pattern_2 = r'[0-9]+ г.п.'                  #year of construction
pattern_3 = r'[а-яА-ЯёЁ]+ р-н'              #District


def extract_info_1(text):
    find_pattern = re.findall(pattern_1, text)
    return find_pattern

def extract_info_2(text):
    find_pattern = re.findall(pattern_2, text)
    return find_pattern

def extract_info_3(text):
    find_pattern = re.findall(pattern_3, text)
    return find_pattern


df_merged['Type'] = df_merged['description'].apply(lambda x: extract_info_1(x))
df_merged['Year'] = df_merged['description'].apply(lambda x: extract_info_2(x))
df_merged['District'] = df_merged['address'].apply(lambda x: extract_info_3(x))

#deleting usefull information
def delete(arr,index):
    while len(arr)>1:
        del arr[index]
    return arr

index_to_delete = 1

df_merged['address']=df_merged['address'].str.replace(r'[а-яА-ЯёЁ]+ р-н, ', '', regex=True)

df_merged['Type'] = df_merged['Type'].apply(lambda x:delete(x,index_to_delete))
df_merged['Year'] = df_merged['Year'].apply(lambda x:delete(x,index_to_delete))


df_merged = df_merged.drop(columns=['main_info', 'description'])
df_merged.to_csv('final_1.csv', index=False)



df_final = pd.read_csv(r"C:\Users\coys7\Apartments_prices\final_1.csv")

df_final = df_final.dropna(axis=0)

#cleaning values
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

df_final[['Floor','Total Floors']]=df_final['floor'].str.split("/", expand=True)

df_final = df_final.dropna(axis=0)

filter=df_final['Type'].isin(["монолитный дом", "панельный дом", "кирпичный дом"])
df_final=df_final[filter]

filter_2=df_final['District'].isin(["Бостандыкский р-н", "Ауэзовский р-н", "Наурызбайский р-н", "Алмалинский р-н", "Алатауский р-н", "Турксибский р-н", "Медеуский р-н", "Жетысуский р-н"])
df_final=df_final[filter_2]


df_final['Floor'] = pd.to_numeric(df_final['Floor'])
df_final['Total Floors'] = pd.to_numeric(df_final['Total Floors'])
df_final['Year'] = pd.to_numeric(df_final['Year'])
df_final['prices'] = pd.to_numeric(df_final['prices'])

df_final = df_final.drop_duplicates()

df_final.to_csv('final_2.csv', index=False)