#!/usr/bin/env python
# coding: utf-8

# # Training Dataset :

# In[1]:


train_data=[
    'This was an awesome movie',
    'Great movie ! I liked it alot',
    'Happy Ending ! awesome acting by the hero',
    'Loved it ! Truly great',
    'bad not upto the mark',
    'Surely a dissapointing movie'
]

y=[1,1,1,1,0,0]


# # Cleaning of a Dataset :

# In[2]:


from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer


# In[3]:


sw=stopwords.words('english')
tokenizer=RegexpTokenizer(r'\w+')
ps=PorterStemmer()


# In[4]:


def cleaning(text):
    text=text.lower()
    text=text.replace('<br /><br />',' ')
    
    # tokenize the text
    data=tokenizer.tokenize(text)
    
    # remove stopwords
    data=[word for word in data if word not in sw]
    
    # stemming
    data=[ps.stem(word) for word in data]
    
    useful_text=' '.join(data)
    return useful_text


# In[5]:


new_training_data=[cleaning(text) for text in train_data]


# In[6]:


new_training_data


# # Vectorizer :

# In[7]:


test_data=['I was happy by seeing the actions in a movie',
          'This movie is bad']

new_test_data=[cleaning(text) for text in test_data]

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[8]:


numbers=cv.fit_transform(new_training_data).toarray()


# In[9]:


# vectorize testing datasets
test_numbers=cv.transform(new_test_data).toarray()
print(test_numbers)


# # Multinomial Naive Bayes :

# In[10]:


from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB


# In[11]:


mnb=MultinomialNB()
mnb.fit(numbers,y)


# In[12]:


mnb.predict(test_numbers)


# # Bernaulli Naive Bayes :

# In[13]:


bnb=BernoulliNB()
bnb.fit(numbers,y)


# In[14]:


bnb.predict(test_numbers)


# In[ ]:




