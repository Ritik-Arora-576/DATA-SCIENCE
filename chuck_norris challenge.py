#!/usr/bin/env python
# coding: utf-8

# In[10]:


import requests
url="http://api.icndb.com/jokes"
data=requests.get(url)


# In[34]:


import json
import numpy as np
data_pair=json.loads(data.content)
data_value=data_pair['value']
A=[]
for i in data_value:
    A.append([i['id'],i['joke']])
A=np.array(A)


# In[36]:


import pandas as pd
df=pd.DataFrame(A,columns=['ID','Joke'])


# In[37]:


df.to_csv('chuck_norris.csv',index=False)


# In[ ]:




