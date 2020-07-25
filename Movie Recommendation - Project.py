#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv('ml-100k/u.data',sep='\t')
df.columns=['user_id','item_id','reviews','timespan']
df.head()


# In[7]:


movie_data=pd.read_csv('ml-100k/u.item',sep='\|',header=None)
movie_data=movie_data[[0,1]]
movie_data.columns=['item_id','title']
movie_data.head()


# In[8]:


review_data=pd.merge(df,movie_data,on='item_id')
review_data.head()


# In[17]:


review_data.groupby('title').mean()['reviews']


# In[18]:


review_data.groupby('title').count()['reviews']


# In[19]:


movie_status=pd.merge(review_data.groupby('title').mean()['reviews'],review_data.groupby('title').count()['reviews'],on='title')
movie_status.columns=['Ratings','Total Reviews']
movie_status.head()


# In[27]:


sns.jointplot(x='Ratings',y='Total Reviews',data=movie_status,color='red',alpha=0.4,kind='scatter')


# In[29]:


movie_pivot_table=review_data.pivot_table(index='user_id',columns='title',values='reviews')


# In[30]:


movie_pivot_table


# In[33]:


star_wars_data=movie_pivot_table['Star Wars (1977)']


# In[56]:


corr_with_starwars=pd.DataFrame(movie_pivot_table.corrwith(star_wars_data))
corr_with_starwars.columns=['Correlations']
corr_with_starwars=corr_with_starwars.dropna()
corr_with_starwars


# In[66]:


highly_recomended=pd.merge(corr_with_starwars,movie_status,on='title').drop(columns=['Ratings']).sort_values(by='Correlations',ascending=False)


# In[67]:


highly_recomended=highly_recomended[highly_recomended['Total Reviews']>100]


# In[68]:


highly_recomended


# In[82]:


def movie_recomendation(movie_name):
    movie=movie_pivot_table[movie_name]
    corr_with_movie=movie_pivot_table.corrwith(movie)
    corr_with_movie.columns=['Correlations']
    corr_with_movie=corr_with_movie.dropna()
    corr_with_movie=pd.DataFrame(corr_with_movie,columns=['Correlations'])
    highly_recomended=corr_with_movie.join(movie_status['Total Reviews']).sort_values(by='Correlations',ascending=False)
    highly_recomended=highly_recomended[highly_recomended['Total Reviews']>100]
    return highly_recomended


# In[87]:


movie_name=input()
movie_recomendation(movie_name)


# In[ ]:




