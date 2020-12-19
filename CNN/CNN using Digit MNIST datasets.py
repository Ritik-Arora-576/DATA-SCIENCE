#!/usr/bin/env python
# coding: utf-8

# ### Classify MNIST datasets:
# 
# - For better accuracy and model we have to build our model deeper , kept kernel size smaller and increases filter layer as we go depper

# In[1]:


from keras import models
from keras.layers import Convolution2D,MaxPooling2D,Dense,Flatten,Dropout


# In[2]:


model=models.Sequential()
model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(64,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(64,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[3]:


model.summary()


# In[4]:


from keras.datasets import mnist
import numpy as np


# In[5]:


(XTrain,YTrain),(XTest,YTest)=mnist.load_data()


# In[6]:


def one_oht(labels):
    classes=len(np.unique(labels))
    examples=labels.shape[0]
    hot=np.zeros((examples,classes))
    for i in range(examples):
        hot[i,labels[i]]=1
    return hot


# In[7]:


XTrain=XTrain.reshape((-1,28,28,1))/255.0
XTest=XTest.reshape((-1,28,28,1))/255.0
YTest=one_oht(YTest)
YTrain=one_oht(YTrain)


# In[8]:


XTrain.shape


# In[9]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[10]:


model.fit(XTrain,YTrain,batch_size=128,epochs=10)


# In[11]:


pred=model.predict(XTest)


# In[12]:


def hot_to_label(pred):
    n=pred.shape[0]
    classes=pred.shape[1]
    pred_label=np.zeros((n,classes))
    for i in range(n):
        pred_label[i,np.argmax(pred[i])]=1
    return pred_label


# In[13]:


pred=hot_to_label(pred)


# In[20]:


pred_=[]
for i in range(10000):
    pred_.append(np.argmax(pred[i]))


# In[23]:


YTest_=[]
for i in range(10000):
    YTest_.append(np.argmax(YTest[i]))


# In[29]:


pred_=np.array(pred_)
YTest_=np.array(YTest_)


# In[31]:


np.sum(YTest_==pred_)/10000


# In[ ]:





# In[ ]:




