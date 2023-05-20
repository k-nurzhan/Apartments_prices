import pandas as pd
import re

df_final = pd.read_csv(r"C:\Users\coys7\Desktop\krisha\app-s_first_page")
df_other_pages = pd.read_csv(r"C:\Users\coys7\Desktop\krisha\app-s_other_pages")
df_final = pd.concat([df_other_pages,df_final])


#splitiing main info into different columns
df_final[['Rooms', 'Square_in_m^2', 'Floor']]=df_final['main_info'].str.split(",", expand=True)


#cleaning data from unnecessary info
df_final['prices']=df_final['prices'].str.replace('[\n ]', '', regex=True)
df_final['address']=df_final['address'].str.replace('[\n]', '', regex=True)
df_final['description']=df_final['description'].str.replace('[\n]', '', regex=True)
df_final['Rooms']=df_final['Rooms'].str.replace('-комнатная квартира', '', regex=True)
df_final['Square_in_m^2']=df_final['Square_in_m^2'].str.replace(' м²', '', regex=True)
df_final['Floor']=df_final['Floor'].str.replace(' этаж', '', regex=True)
df_final['prices']=df_final['prices'].str.replace('от', '', regex=True)
df_final['address']=df_final['address'].str.replace('   ', '', regex=True)
df_final['address']=df_final['address'].str.replace('["]', '', regex=True)

#working with description
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


df_final['Type'] = df_final['description'].apply(lambda x: extract_info_1(x))
df_final['Year'] = df_final['description'].apply(lambda x: extract_info_2(x))
df_final['District'] = df_final['address'].apply(lambda x: extract_info_3(x))



def delete(arr,index):
    while len(arr)>1:
        del arr[index]
    return arr
    #else:
     #   return arr
index_to_delete = 1

df_final['address']=df_final['address'].str.replace(r'[а-яА-ЯёЁ]+ р-н, ', '', regex=True)



df_final['Type'] = df_final['Type'].apply(lambda x:delete(x,index_to_delete))
df_final['Year'] = df_final['Year'].apply(lambda x:delete(x,index_to_delete))

df_final = df_final.drop(columns=['main_info', 'description'])
df_final.to_csv('final_21', index=False)