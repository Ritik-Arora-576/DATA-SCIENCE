#!/usr/bin/env python
# coding: utf-8

# # PANDAS :

# - Pandas is use to represent data in tabular form i.e in the form of rows and column

# In[2]:


import pandas as pd
import numpy as np
user_data={
    'MarksA':np.random.randint(1,100,5),
    'MarksB':np.random.randint(50,100,5),
    'MarksC':np.random.randint(1,100,5),
}
print(user_data)


# In[7]:


df=pd.DataFrame(user_data,dtype=np.float64)
print(df)


# In[8]:


# print table
df.head(3)


# In[10]:


print(df.columns)


# In[44]:


# creating csv:
# Index =False means u dont want to add index in csv file
df.to_csv('pandas.csv',index=False)


# In[15]:


x=pd.read_csv('pandas.csv')
# if we want to delete particular column
x=x.drop(columns='Unnamed: 0')
x.head()


# In[16]:


x.describe()


# In[18]:


# print last 2 rows of tabualar data
x.tail(2)


# In[20]:


x=[['k,d','c,d',3],[4,5,6],[6,7,8]]
df=pd.DataFrame(x,columns=['Marks 1','Marks 2','Marks 3'])
df.head()


# In[23]:


print(df.iloc[1,1:])


# In[22]:


df.iloc[1][1]


# In[30]:


# in order to get the location of Marks 2
df.columns.get_loc('Marks 2')


# In[31]:


x=[[70,80,45],[89,56,66],[99,43,76]]
df=pd.DataFrame(x,columns=['Maths','Physics','Chemistry'])
df.head()


# In[32]:


df.describe()


# In[37]:


# sort values in pandas :
df.sort_values(by=["Maths","Physics"],ascending=False)


# In[38]:


get_ipython().set_next_input('data_values=df.values');get_ipython().run_line_magic('pinfo', 'df.values')


# In[46]:


# df.values is used to convert data frames into numpy array
data_values=df.values
print(type(df))
print(type(data_values))


# In[1]:


print(data_values)
print(data_values.shape)


# ### FUNCTIONS IN PANDAS :
# - pd.DataFrame() (use to frame 2-D data in tabular Form)
# - df.head() (Print tabular Data)
# - df.describe() (Describe about tabular data)
# - df.tail() (print tabular Data from last Rows)
# - df.values (use to convert tabular data into numpy array)
# - df.sort_values() (sort values by given conditions in parameter)
# - df.iloc[][] (use to print data of particular index)
# - df.drop() (delete a particular column)

# - df.to_csv() (convert tabular data into csv file)

# In[ ]:




