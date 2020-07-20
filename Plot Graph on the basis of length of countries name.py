#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
url="https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
data=requests.get(url)


# In[5]:


from bs4 import BeautifulSoup as soup
data_soup=soup(data.content,'lxml')


# In[35]:


tables=data_soup.find_all('table',{'class':'wikitable sortable'})
first_table=tables[0]


# In[36]:


headings=first_table.find_all('th')
list_of_headings=[]
for h in headings:
    list_of_headings.append(h.text)
print(list_of_headings)


# In[45]:


table_data=[]
rows=first_table.find_all('tr')
for row in rows:
    current_data=[]
    datas=row.find_all('td')
    for idx,data in enumerate(datas):
        if idx==1:
            current_data.append(data.text[1:].replace(',',''))
        elif idx==2:
            current_data.append(data.text.replace(',',''))
        else:
            current_data.append(data.text)
    table_data.append(current_data)
table_data=table_data[2:]


# In[47]:


file_name='GDP.csv'
with open(file_name,'w',encoding='UTF-8') as f:
    f.write(','.join(list_of_headings))
    for data in table_data:
        f.write(','.join(data))


# In[63]:


import pandas as pd
df=pd.read_csv(file_name)
all_countries=[]
for i in df.get('Country/Territory'):
    i=i.split('[')[0]
    i=i.strip()
    all_countries.append(i)


# In[95]:


dictionary={}
for country in all_countries:
    length=len(country)
    if length in dictionary:
        dictionary[length]+=1
    else:
        dictionary[length]=1
sorted_dictionary={}
for i in sorted(dictionary):
    sorted_dictionary[i]=dictionary[i]
dictionary=sorted_dictionary


# In[96]:


import numpy as np
x_cordinate=np.array(list(dictionary.keys()))
y_cordinate=np.array(list(dictionary.values()))


# In[99]:


import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
plt.style.use('dark_background')
plt.plot(x_cordinate,y_cordinate,color='green',marker='o')
plt.xlabel('Number of words in county name')
plt.ylabel('Number of countries')
plt.title('Probablity Distribution')
plt.ylim(0,50)
plt.xlim(0,35)
plt.show()


# In[101]:


plt.bar(x_cordinate,y_cordinate)


# In[ ]:




