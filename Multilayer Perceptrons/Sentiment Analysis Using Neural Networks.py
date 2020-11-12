#!/usr/bin/env python
# coding: utf-8

# ## DATA PREPERATION :

# In[4]:


import numpy as np
from keras.datasets import imdb


# In[5]:


((XT,YT),(Xt,Yt))=imdb.load_data(num_words=10000) # a sentence contain maximum words upto 10000


# In[16]:


print(XT[0]) # XT[0] is in the form of list of numbers where each number is mapped with a word


# In[7]:


word_idx=imdb.get_word_index()


# In[8]:


print(word_idx)


# In[9]:


idx_word=dict([word_idx[i],i] for i in word_idx)


# In[11]:


print(idx_word)


# In[14]:


str=''
for i in XT[1]:
    str+=idx_word[i]+' '


# In[15]:


str


# In[19]:


# vectorize the given data
def vectorize(sentences,max_words):
    m=sentences.shape[0]
    vectorize_vector=np.zeros((m,max_words))
    for i,idx in enumerate(sentences):
        vectorize_vector[i,idx]=1
        
    return vectorize_vector


# In[20]:


# convert the given data into [011000110......] format

X_train=vectorize(XT,10000)
X_test=vectorize(Xt,10000)
Y_train=np.array(YT,dtype=np.float64)
Y_test=np.array(Yt,dtype=np.float64)


# In[21]:


X_train.shape


# ## Define your model architechture :

# In[53]:


from keras import models
from keras.layers import Dense


# In[54]:


# creating a neural network

model=models.Sequential()
model.add(Dense(16,activation='relu',input_shape=(10000,))) # Creating first hidden layer of 16 units
model.add(Dense(16,activation='relu')) # Creating second hidden layer of 16 units
model.add(Dense(1,activation='sigmoid')) # Creating output layer of 1 unit


# In[55]:


# compile or model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[56]:


model.summary()


# ## Train our model :

# In[57]:


x_val=X_train[:5000,:]
x_train_new=X_train[5000:,:]

y_val=Y_train[:5000]
y_train_new=Y_train[5000:]


# In[36]:


hist=model.fit(x_train_new,y_train_new,batch_size=512,epochs=20,validation_data=(x_val,y_val))


# In[37]:


import matplotlib.pyplot as plt


# In[43]:


h=hist.history


# ## Visualize losses and accuracy :

# In[48]:


plt.style.use('seaborn')
plt.plot(h['val_loss'],label='Validation Loss',color='orange')
plt.plot(h['loss'],label='Loss',color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[51]:


plt.style.use('seaborn')
plt.plot(h['val_accuracy'],label='Validation Accuracy',color='orange')
plt.plot(h['accuracy'],label='Accuracy',color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# - Stop Our algorithm by 3 epochs because after 3 epochs cross validation accuracy starts to decrease

# In[58]:


hist=model.fit(x_train_new,y_train_new,batch_size=512,epochs=4,validation_data=(x_val,y_val))

h=hist.history

plt.style.use('seaborn')
plt.plot(h['val_loss'],label='Validation Loss',color='orange')
plt.plot(h['loss'],label='Loss',color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.style.use('seaborn')
plt.plot(h['val_accuracy'],label='Validation Accuracy',color='orange')
plt.plot(h['accuracy'],label='Accuracy',color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ### Test our data for testing data

# In[62]:


# returns the accuracy of our testing data
model.evaluate(X_test,Y_test)[1]


# In[ ]:




