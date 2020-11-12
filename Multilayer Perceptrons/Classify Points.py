#!/usr/bin/env python
# coding: utf-8

# ## Data Preperation :

# In[27]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[28]:


dfx=pd.read_csv('Logistic_X_Train.csv')
dfy=pd.read_csv('Logistic_Y_Train.csv')


# In[41]:


x_train=np.array(dfx)
y_train=np.array(dfy,dtype=np.float64).reshape((-1,))


# In[44]:


split=int(0.75*x_train.shape[0])
x_new_train=x_train[:split,:]
x_val=x_train[split:,:]
y_new_train=y_train[:split]
y_val=y_train[split:]


# In[49]:


plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
plt.show()


# In[50]:


df=pd.read_csv('Logistic_X_Test.csv')
x_test=np.array(df)


# ## Fit our data into our model :

# In[78]:


from keras import models
from keras.layers import Dense


# In[79]:


model=models.Sequential()
model.add(Dense(10,activation='relu',input_shape=(2,)))
model.add(Dense(5,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[80]:


model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')


# In[81]:


hist=model.fit(x_new_train,y_new_train,batch_size=256,epochs=500,validation_data=(x_val,y_val))


# In[82]:


print(hist.history)


# In[83]:


plt.plot(hist.history['val_loss'],label='Validation Loss',color='blue')
plt.plot(hist.history['loss'],label='Loss',color='red')
plt.legend()
plt.show()


# In[84]:


plt.plot(hist.history['val_accuracy'],label='Validation Accuracy',color='blue')
plt.plot(hist.history['accuracy'],label='Accuracy',color='red')
plt.legend()
plt.show()


# In[88]:


pred=model.predict(x_test).reshape((-1,))


# In[89]:


pred.shape


# In[90]:


y_test=[]
for i in range(pred.shape[0]):
    if pred[i]>0.5:
        y_test.append(1)
    else:
        y_test.append(0)


# In[92]:


y_test=np.array(y_test)


# In[93]:


print(y_test)


# In[95]:


df=pd.DataFrame(data=y_test,columns=['label'])


# In[98]:


df.to_csv('classify_points.csv',index=False)


# In[ ]:




