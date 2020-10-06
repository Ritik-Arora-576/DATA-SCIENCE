#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np


# In[35]:


df=pd.read_csv('Train.csv')
df


# In[36]:


columns_to_drop=['name','ticket','cabin','embarked','boat','body','home.dest']
df_cleaned=df.drop(columns=columns_to_drop)


# In[37]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_cleaned['sex']=le.fit_transform(df_cleaned['sex'])


# In[56]:


mean_fare=df_cleaned['fare'].mean()
print(mean_fare)


# In[52]:


df_cleaned=df_cleaned.fillna(df_cleaned['age'].mean(),axis=0)


# In[64]:


df_cleaned


# In[63]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy',max_depth=5)


# In[66]:


input_col=['pclass','sex','age','sibsp','parch','fare']
output_col=['survived']

x=df_cleaned[input_col]
y=df_cleaned[output_col]


# In[70]:


dtc.fit(x,y)


# In[72]:


testing=pd.read_csv('Test.csv')
cleaned_testing=testing.drop(columns=columns_to_drop)


# In[80]:


cleaned_testing=cleaned_testing.fillna(df_cleaned['age'].mean(),axis=0)
cleaned_testing['sex']=le.fit_transform(cleaned_testing['sex'])


# In[84]:


pred=dtc.predict(cleaned_testing)


# In[94]:


pred=np.array(pred,dtype=np.int8)


# In[99]:


pred=pred.reshape((-1,1))


# In[100]:


id=np.arange(300)
id=np.array(id).reshape((-1,1))


# In[103]:


pred=np.concatenate((id,pred),axis=1)


# In[105]:


df=pd.DataFrame(pred,columns=['Id','survived'])


# In[107]:


df.to_csv('titanic_data.csv',index=False)


# In[ ]:




