#!/usr/bin/env python
# coding: utf-8

# # SEABORN :

# In[2]:


import seaborn as sns
import numpy as np
tips=sns.load_dataset('tips')
tips.head(30)


# In[18]:


# plot the bar graph of average total bills by each sex
sns.barplot(x='sex',y='total_bill',data=tips)


# In[19]:


# Plot the graph of standard deviation of total_bills
sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)


# In[21]:


# Plot the count of number of males and females 
sns.countplot(x='sex',data=tips)


# In[27]:


sns.countplot(x='day',data=tips)


# In[25]:


sns.boxplot(x='day',y='total_bill',data=tips)


# In[28]:


sns.boxplot(x='day',y='total_bill',data=tips,hue='sex')


# In[31]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex')


# In[32]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex',split=True)


# In[47]:


# Distribution plot
sns.distplot(tips['total_bill'],color='red',kde=False,bins=30)


# In[49]:


sns.kdeplot(tips['total_bill'],color='green')


# In[61]:


# Joint plot is used to determine the relation between two parameters
sns.jointplot(x='total_bill',y='tip',data=tips,kind='kde',color='pink')


# In[63]:


sns.pairplot(tips,hue='sex')


# In[3]:


flights=sns.load_dataset('flights')
flights.head()


# In[4]:


flights.corr()


# In[6]:


sns.heatmap(flights.corr(),annot=True)


# In[7]:


get_ipython().set_next_input('pivot_tables=flights.pivot_table');get_ipython().run_line_magic('pinfo', 'flights.pivot_table')


# In[13]:


pivot_tables=flights.pivot_table(columns='year',index='month',values='passengers')
pivot_tables


# In[20]:


sns.heatmap(pivot_tables,linecolor='white')


# In[21]:


titanic=sns.load_dataset('titanic')
titanic.head()


# In[27]:


pivot_table=titanic.pivot_table(columns='pclass',index='sex',values='fare')
pivot_table


# In[28]:


sns.heatmap(pivot_table)


# In[ ]:




