#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[8]:


dfx=pd.read_csv('Diabetes_XTrain.csv')
dfy=pd.read_csv('Diabetes_YTrain.csv')
x_values=dfx.values
y_values=dfy.values.reshape(-1,)


# In[28]:


testing_data=pd.read_csv('Diabetes_Xtest.csv')
test_values=testing_data.values


# In[12]:


def distance(x1,x2):
    return np.sqrt(sum((x2-x1)**2))

def knn(x_data,y_data,query,m=5):
    val=[]
    total_points=y_data.shape[0]
    for i in range(total_points):
        val.append((distance(x_data[i],query),y_data[i]))
        
    val=sorted(val)
    val=np.array(val[:m])
    uni=np.unique(val[:,1],return_counts=True)
    index=uni[1].argmax()
    return uni[0][index]


# In[27]:


values=[]
for i in range(test_values.shape[0]):
    prediction=int(knn(x_values,y_values,test_values[i]))
    values.append(prediction)
    
df=pd.DataFrame(values,columns=['Outcome'])
df.to_csv('Diabetes_challenge.csv',index=False)


# In[25]:


get_ipython().run_line_magic('pinfo', 'df.to_csv')


# In[ ]:




