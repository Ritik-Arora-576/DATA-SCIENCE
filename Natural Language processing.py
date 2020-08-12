#!/usr/bin/env python
# coding: utf-8

# # Get the Data/Corpus :

# In[2]:


from nltk.corpus import brown
print(brown.categories())


# In[3]:


data=brown.sents(categories='fiction')
print(data)
print(len(data))


# In[4]:


print(' '.join(data[15]))


# # Tokenization and Stopward removal :

# In[5]:


# sent_tokenize = break the document into sentence ........
# word_tokenize = break the sentence into word ............


from nltk.tokenize import sent_tokenize , word_tokenize

document='''I am Ritik Arora . Study in Delhi Technological University . 
Recently passionate about Data Science .'''

sentences=sent_tokenize(document)
print(sentences)
print(len(sentences))


# In[6]:


sentence='You can mail me on my gmail ID ritikarora656@gmail.com'
words=word_tokenize(sentence)
print(words)
print(sentence.split())


# ### Stopward Removal :
# 
# -Use to remove non-meaningfull words from the sentence

# In[7]:


from nltk.corpus import stopwords

# sw contains sets of words which can be ignore in the sentence
sw=set(stopwords.words('english'))
print(sw)


# In[8]:


def remove_stopwards(texts,stopwards):
    useful_words=[i for i in texts if i not in stopwards]
    return useful_words

text = 'i loved that movie so much'.split()
useful_words=remove_stopwards(text,sw)
print(useful_words)


# ### Regular Expression Tokenizer :

# In[9]:


from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer('[a-zA-Z@.]+')
text='hello I am @Ritik Arora .'
useful_text=tokenizer.tokenize(text)
print(useful_text)


# # STEMMING :

# - It is used to convert similar words like ( jumps , jumping , jumps , jumped ==> jump )

# In[10]:


from nltk.stem.snowball import SnowballStemmer , PorterStemmer


# In[11]:


ps=PorterStemmer()


# In[12]:


ps.stem('lovely')


# In[13]:


ps.stem('jumping')


# In[14]:


ps.stem('calling')


# In[15]:


# basic difference between porter stemmer and snowball stemmer is:
# - porter stemmer is used for English language only while snowball stemmer is used for multilanguage like French,german,italian
   # ,engilsh etc


# In[16]:


ss=SnowballStemmer('english')
ss.stem('loving')


# In[17]:


ss.stem('jumping')


# # Build A bag and Vectorization :

# In[18]:


corpus=[
    'Indian Cricket Team will win world cup, say captain Virat Kohli . World Cup will be held at Sri Lanka',
    'We will win next Lok Sabha election , said by Indian PM',
    'The nobel laurate won the heart of many peaple',
    'The movie Raazi was an exciting thriller spy based on the thriller story'
]
# corpus contains 4 sentences having different categories :
# 1. sports  2. Politics  3. Economics  4. movie


# In[19]:


# convert a text into array of numbers
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
vectorized_corpus=cv.fit_transform(corpus)


# In[20]:


vectorized_corpus=vectorized_corpus.toarray()


# In[21]:


print(vectorized_corpus[0])


# In[22]:


# it shows us the dictionary of words and thier corresponding values

print(cv.vocabulary_)


# In[23]:


# bag of words 

numbers=vectorized_corpus[2]
cv.inverse_transform(numbers)


# # Remove stopwords using vectorizer :

# In[31]:


def mytokenizer(sentence):
    words=tokenizer.tokenize(sentence.lower())
    
    # remove stopwords
    words=remove_stopwards(words,sw)
    return words

sentence='i sent this chapter related document to ritikarora656@gmail.com'
mytokenizer(sentence)


# In[32]:


# so it tokenize the sentence into words and remove meaningless words from that sentence

cv=CountVectorizer(tokenizer=mytokenizer)


# In[34]:


numbers=cv.fit_transform(corpus).toarray()
print(numbers)


# In[35]:


print(cv.vocabulary_)


# In[36]:


print(cv.inverse_transform(numbers))


# - for testing data we have to use transform method rather than fit transform data in order to avoid overwriting of data

# In[37]:


testing_data=['India will win the World Cup']
numbers=cv.transform(testing_data).toarray()
print(cv.vocabulary_)


# In[38]:


cv.inverse_transform(numbers)


# # Uni-gram, Bi-gram , Tri-gram and n-gram features :

# In[42]:


cv=CountVectorizer(ngram_range=(2,2)) #bi-gram features
# (3,3) for tri-gram feature
# (1,3) for uni-gram , bi-gram and tri-gram features


# In[43]:


new_corpus=[
    'This is a good movie',
    'This is a good movie but songs are not good',
    'This is not a good movie'
]


# In[44]:


numbers=cv.fit_transform(new_corpus).toarray()
print(numbers)


# In[47]:


print(cv.vocabulary_) # pair two consequtive word together


# # Tf-idf Normalisation :
# - this is use to avoid feature/word which is used more frequently across different sets of Document
# - For this it is used to assign weights to different text . If word repeats more frequently across documnets then its weight will be less
# - Formula = log(N/count(t,d))  where N is number of Documents and count(t,d) is frequency of a word repeated

# In[1]:


sent1='This movie is good'
sent2='That movie was so good'
sent3='This movie is not good'

corpus=[sent1,sent2,sent3]


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfdi=TfidfVectorizer()
weights=tfdi.fit_transform(corpus).toarray()
print(weights)


# In[5]:


print(tfdi.vocabulary_)


# In[ ]:




