#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2


# ### DATA PREPERATION:

# In[14]:


dataset=pd.read_csv('fashion-mnist_train.csv')


# In[15]:


dataset=np.array(dataset)


# In[16]:


X_train=dataset[:,1:]
y_train=dataset[:,0]


# In[17]:


def one_oht(data):
    
    classes=len(np.unique(data))
    hot=np.zeros((data.shape[0],classes))
    for i in range(data.shape[0]):
        hot[i,data[i]]=1
        
    return hot


# In[19]:


X_train=X_train.reshape((-1,28,28,1))
y_train=one_oht(y_train)


# In[20]:


y_train


# ### Preparing our CNN model:
# 

# In[22]:


from keras import models
from keras.layers import Dense,Convolution2D,MaxPooling2D,Dropout,Flatten


# In[27]:


model=models.Sequential()
model.add(Convolution2D(64,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(32,(5,5),activation='relu'))
model.add(Convolution2D(8,(5,5),activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))


# In[28]:


model.summary()


# In[30]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[32]:


hist=model.fit(X_train,y_train,batch_size=512,epochs=5,validation_split=0.20)


# In[38]:


plt.style.use('seaborn')
plt.plot(hist.history['accuracy'],label='Accuracy')
plt.plot(hist.history['val_accuracy'],label='Validation Accuracy')
plt.plot(hist.history['val_loss'],label='Validation Loss')
plt.plot(hist.history['loss'],label='loss')
plt.legend()
plt.show()


# ### Predicting our testing data:

# In[39]:


dataset=pd.read_csv('fashion-mnist_test.csv')


# In[41]:


dataset=np.array(dataset)


# In[42]:


X_test=dataset[:,1:]
y_test=dataset[:,0]


# In[46]:


X_test=X_test.reshape((-1,28,28,1))


# In[47]:


pred=model.predict(X_test)


# In[50]:


labels=[]

for i in range(pred.shape[0]):
    labels.append(np.argmax(pred[i]))


# In[54]:


labels=np.array(labels)


# In[57]:


np.sum(labels==y_test)/10000


# In[ ]:




