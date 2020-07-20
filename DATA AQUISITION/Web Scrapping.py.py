#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib.request import urlopen
url="https://samples.openweathermap.org/data/2.5/weather?q=London,uk&appid=439d4b804bc8187953eb36d2a8c26a02"
data=urlopen(url)
url_html=data.read()
print(url_html)


# # JSON :-

# In[6]:


import json
json_data=json.loads(url_html)
print(json_data)
print(type(json_data)) # Json makes the following data in the form of dictionary(key,value) pairs
print(json_data['coord'])


# In[8]:


str_json=json.dumps(json_data)
print(type(str_json))
print(str_json)


# ## 2. GOOGLE API s:

# In[1]:


import requests
url='http://maps.googleapis.com/maps/api/geocode/json?'
parameters ={
    'address':'coding blocks pitampura',
    'key':''
}
r=requests.get(url,params=parameters)
r.url
print(r.content)
print(r.content_decode("UTF-8"))


# ## Facebook API:

# In[10]:


import requests
url="http://graph.facebook.com/4/picture?type=large"
r=requests.get(url)
print(r.content)


# In[7]:


with open("facebook.jpg",'wb') as f:
    f.write(r.content)

