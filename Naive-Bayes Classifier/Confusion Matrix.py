#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.naive_bayes import MultinomialNB,GaussianNB


# In[2]:


mnb=MultinomialNB()
gnb=GaussianNB()


# In[4]:


from sklearn.datasets import make_classification


# In[6]:


X,y=make_classification(n_samples=200,n_features=2,n_informative=2,n_redundant=0,random_state=2)


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


# In[9]:


gnb.fit(X,y)


# In[13]:


y_pred=gnb.predict(X)


# In[14]:


gnb.score(X,y)


# In[17]:


from sklearn.metrics import confusion_matrix


# In[18]:


cnf=confusion_matrix(y_pred,y)


# In[19]:


cnf


# In[23]:


import seaborn as sns
sns.heatmap(cnf)
plt.legend()
plt.show()


# In[21]:


get_ipython().run_line_magic('pinfo', 'sns.heatmap')


# In[ ]:




