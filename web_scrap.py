from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

def first_page():
    web = f'https://krisha.kz/prodazha/kvartiry/almaty/?das[_sys.hasphoto]=1&das[who]=1'
    response = requests.get(web)
    content = response.text
    soup = BeautifulSoup(content, 'lxml')
    prices = []
    main_info = []
    address = []
    description = []

    Title = soup.find_all('div', class_= 'a-card__descr')

    for main in Title:
        prices.append(main.find('div', 'a-card__price').get_text())
        main_info.append(main.find('a', 'a-card__title').get_text())
        address.append(main.find('div', 'a-card__subtitle').get_text())
        description.append(main.find('div', 'a-card__text-preview').get_text())
    
    dict_appartments = {'main_info': main_info, 'address': address, 'description': description, 'prices': prices}
    df_appartments = pd.DataFrame(dict_appartments)
    return df_appartments

first = first_page()
first.to_csv('app-s_first_page', index=False)

prices_other = []
main_info_other = []
address_other = []
description_other = []

def remaining_pages():
    for pages in range(2, 863):
        web = f'https://krisha.kz/prodazha/kvartiry/almaty/?das[_sys.hasphoto]=1&das[who]=1&page={pages}'
        response = requests.get(web)
        content = response.text
        soup = BeautifulSoup(content, 'lxml')
        

        Title = soup.find_all('div', class_= 'a-card__descr')

        for main in Title:
            prices_other.append(main.find('div', 'a-card__price').get_text())
            main_info_other.append(main.find('a', 'a-card__title').get_text())
            address_other.append(main.find('div', 'a-card__subtitle').get_text())
            description_other.append(main.find('div', 'a-card__text-preview').get_text())
        
        dict_appartments = {'main_info': main_info_other, 'address': address_other, 'description': description_other, 'prices': prices_other}
        df_appartments = pd.DataFrame(dict_appartments)
        time.sleep(10)
    
    return df_appartments

other_pages = remaining_pages()
other_pages.to_csv('app-s_other_pages', index=False)