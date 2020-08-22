#!/usr/bin/env python
# coding: utf-8

# ### Preparing Datasets :

# In[1]:


import pandas as pd


# In[17]:


df=pd.read_csv('Train.csv')
df.head(15)


# # Data Cleaning :

# In[3]:


from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords 


# In[70]:


tokenizer=RegexpTokenizer('[a-zA-Z]+')
ps=PorterStemmer()
sw=set(stopwords.words('english'))


# In[71]:


def data_cleaning(text):
    text=text.lower()
    text=text.replace('<br /><br />',' ')
    
    data=tokenizer.tokenize(text)
    
    data=[word for word in data if word not in sw]
    data=[ps.stem(word) for word in data]
    
    text=' '.join(data)
    return text


# In[72]:


cleaned_review=[]
total=df.shape[0]

for i in range(total):
    cleaned_review.append(data_cleaning(df.iloc[i,0]))


# In[73]:


labels=[]
for i in range(total):
    if df.iloc[i,1]=='pos':
        labels.append(1)
    else:
        labels.append(0)


# In[74]:


import numpy as np
cleaned_review=np.array(cleaned_review).reshape((-1,1))
labels=np.array(labels).reshape((-1,1))


# In[75]:


cleaned_review=np.concatenate((cleaned_review,labels),axis=1)


# In[76]:


new_df=pd.DataFrame(cleaned_review,columns=['review','label'])


# In[77]:


new_df


# In[78]:


testing_df=pd.read_csv('Test.csv')


# In[79]:


cleaned_test_reviews=[]
total=testing_df.shape[0]
for i in range(total):
    cleaned_test_reviews.append(data_cleaning(testing_df.iloc[i,0]))


# In[80]:


test_df=pd.DataFrame(cleaned_test_reviews,columns=['review'])
test_df


# ### Vectorize the data :

# In[81]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[82]:


train_numbers=cv.fit_transform(new_df.iloc[:,0])


# In[84]:


test_numbers=cv.transform(test_df.iloc[:,0])


# # Prediction :

# In[85]:


from sklearn.naive_bayes import BernoulliNB


# In[86]:


bnb=BernoulliNB()


# In[87]:


bnb.fit(train_numbers,new_df.iloc[:,1])


# In[88]:


prediction=bnb.predict(test_numbers)


# In[89]:


print(prediction)


# In[91]:


pred=[]
for i in range(prediction.shape[0]):
    if prediction[i]=='1':
        pred.append('pos')
        
    else:
        pred.append('neg')


# In[97]:


pred=np.array(pred).reshape((-1,1))
ID=np.arange(pred.shape[0]).reshape((-1,1))


# In[99]:


final_data=np.concatenate((ID,pred),axis=1)


# In[102]:


df=pd.DataFrame(final_data,columns=['Id','label'])


# In[104]:


df.to_csv('movie_reviews_classifier.csv',index=False)


# In[ ]:




