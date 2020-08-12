#!/usr/bin/env python
# coding: utf-8

# - for very high dimensionality the accuracy of a model get decreases
# - In order to increase the accuracy of a model we have to choose perfect subset of features whose contribution in a model is relatively high

# In[5]:


import numpy as np
import pandas as pd


# In[6]:


from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


# In[7]:


df=pd.read_csv('data/datasets_train.csv')
df


# In[8]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[9]:


best_features=SelectKBest(score_func=chi2,k=10)


# In[10]:


fit=best_features.fit(X,y)


# In[11]:


dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(X.columns)


# In[12]:


df=pd.concat((dfscores,dfcolumns),axis=1)
df.columns=['Scores','Features']
df.sort_values(by='Scores',ascending=False)


# In[13]:


import matplotlib.pyplot as plt
plt.bar(df['Features'],df['Scores'])
plt.show()


# # Feature Importance :

# In[14]:


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# In[15]:


model=RandomForestClassifier()
model.fit(X,y)


# In[17]:


model.feature_importances_


# In[20]:


df=pd.DataFrame(model.feature_importances_,index=X.columns,columns=['Importance']).sort_values(by='Importance',ascending=False)


# In[24]:


df


# In[27]:


plt.figure(figsize=(20,10))
plt.style.use('seaborn')
plt.bar(df.index,df['Importance'],color='red')
plt.show()


# # Correlation Matrix :-

# In[29]:


X.corr()


# In[30]:


import seaborn as sns


# In[38]:


plt.figure(figsize=(20,20))
sns.heatmap(X.corr(),annot=True)


# In[36]:


get_ipython().run_line_magic('pinfo', 'sns.heatmap')


# In[ ]:




