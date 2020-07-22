#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup as soup
import pandas as pd


# In[12]:


product_information=[]
def extract_information(url,i):
    data=requests.get(url)
    data_soup=soup(data.content,'lxml')
    all_products=data_soup.find_all('li')
    for product in all_products:
        try:
            img_src=product.find('img').attrs['src']
            book_name=product.find('h3')
            book_name=book_name.find('a').attrs['title']
            price=product.find('p',{'class':'price_color'}).text
            product_information.append([img_src,book_name,price])
        except Exception as e:
            pass
    #next_button=data_soup.find('li',{'class':'next'})
   # print(next_button)
    if i<50:
        url="http://books.toscrape.com"+'/'+('catalogue/page-{}.html'.format(i+1))
        extract_information(url,i+1)


# In[13]:


url="http://books.toscrape.com"
extract_information(url,1)


# In[14]:


df=pd.DataFrame(product_information,columns=['image_url','book_title','product_price'])
df.to_csv('book_scrap.csv',index=False)


# In[ ]:




