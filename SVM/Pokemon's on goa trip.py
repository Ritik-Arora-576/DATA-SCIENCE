#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path 
from keras.preprocessing import image


# ### Data Prepration :

# In[45]:


p=Path('Pokemon_images')

img_list=p.glob('*.jpg')

image_data=[]

for img_path in img_list:
    label=str(img_path).split('\\')[-1]
    img=image.load_img(img_path,target_size=(32,32))
    img_array=image.img_to_array(img)
    image_data.append(img_array)


# In[46]:


image_data=np.array(image_data,dtype='float32')/255.0


# In[47]:


plt.imshow(image_data[76])
plt.show()


# In[48]:


label_dict={'Pikachu':0 , 'Bulbasaur':1 , 'Charmander':2}


# In[49]:


df=pd.read_csv('Pokemon_train.csv')
name_of_pokemon=df['NameOfPokemon']


# In[50]:


labels=[]
for name in name_of_pokemon:
    labels.append(label_dict[name])


# In[51]:


labels=np.array(labels)


# In[52]:


image_data=image_data.reshape((image_data.shape[0],-1))


# In[53]:


image_data.shape


# In[103]:


p=Path('Pokemon_test_images')
image_list=p.glob('*.jpg')

test_image_data=[]
test_labels=[]

for img_path in image_list:
    label=str(img_path).split('\\')[-1]
    test_labels.append(label)
    img=image.load_img(img_path,target_size=(32,32))
    img_array=image.img_to_array(img)
    test_image_data.append(img_array)


# In[106]:


test_image_data=np.array(test_image_data,dtype='float32')/255.0
test_labels=np.array(test_labels)
test_image_data=test_image_data.reshape((test_image_data.shape[0],-1))


# ### Fit our training data into our model :

# In[66]:


from sklearn import svm
svc=svm.SVC()


# In[69]:


params={
    'kernel':['linear','rbf','sigmoid','poly'],
    'C': [0.1,0.2,0.5,1.0,2.0,5.0,7.0,10.0]
}
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(svc,params,'accuracy',8,cv=10)
gs.fit(image_data,labels)


# In[70]:


gs.best_estimator_


# In[71]:


gs.best_score_


# In[94]:


svc=svm.SVC(C=5.0,kernel='rbf')


# In[95]:


svc.fit(image_data,labels)


# In[96]:


from sklearn.model_selection import cross_val_score
cross_val_score(svc,image_data,labels,cv=10,scoring='accuracy').mean()


# ### Predict our testing data :

# In[111]:


pred=svc.predict(test_image_data)


# In[112]:


inv_label_dict={0:'Pikachu' , 1:'Bulbasaur' , 2:'Charmander'}


# In[115]:


pred


# In[124]:


pred=pred.reshape((pred.shape[0],1))
test_labels=test_labels.reshape((test_labels.shape[0],1))


# In[125]:


zipped=np.concatenate((test_labels,pred),axis=1)


# In[157]:


df=pd.DataFrame(zipped,columns=['ImageId','NameOfPokemon'])


# In[158]:


df


# In[138]:


test=pd.read_csv('Pokemon_test.csv')


# In[140]:


test=test['ImageId']


# In[155]:


test=np.array(test)
test_labels=test_labels.reshape((test_labels.shape[0],))
pred=pred.reshape((pred.shape[0],))


# In[159]:


test_pred=[]
for image_id in test:
    j=0
    for idx in test_labels:
        if idx==image_id:
            test_pred.append(pred[j])
            break
        j+=1


# In[161]:


test_pred=np.array(test_pred)
test_pred=test_pred.reshape((test_pred.shape[0],1))
test=test.reshape((test.shape[0],1))


# In[164]:


zipped=np.concatenate((test,test_pred),axis=1)


# In[165]:


df=pd.DataFrame(zipped,columns=['ImageId','NameOfPokemon'])


# In[167]:


test_pred=test_pred.reshape((test_pred.shape[0],))


# In[177]:


testing=[]
for j in test_pred:
    testing.append(inv_label_dict[j])


# In[182]:


testing=np.array(testing)
testing=testing.reshape((testing.shape[0],1))


# In[188]:


zipped=np.concatenate((test_labels,testing),axis=1)


# In[189]:


df=pd.DataFrame(zipped,columns=['ImageId','NameOfPokemon'])


# In[190]:


df


# In[191]:


df.to_csv('Pokemon.csv',index=False)


# In[ ]:




