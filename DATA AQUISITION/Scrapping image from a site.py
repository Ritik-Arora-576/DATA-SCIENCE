#!/usr/bin/env python
# coding: utf-8

# In[9]:


import requests
import bs4
url="https://home.howard.edu"
data=requests.get(url)
print(data.content)


# In[10]:


soup=bs4.BeautifulSoup(data.text,'html.parser')
print(soup)


# In[11]:


all_images=soup.find_all('img')


# In[21]:


x=1
for i in all_images:
    if '.jpg' in i.attrs['src']:
        with open('howard{}.jpg'.format(x),'wb') as f:
            if i.attrs['src'][0]=='/':
                data=requests.get(url+i.attrs['src'])
                f.write(data.content)
            else:
                data=requests.get(i.attrs['src'])
                f.write(data.content)
    x+=1


# In[13]:





# In[ ]:




