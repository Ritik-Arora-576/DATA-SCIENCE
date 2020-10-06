#!/usr/bin/env python
# coding: utf-8

# ## Data Preprocessing :

# In[1]:


import numpy as np
import os
from pathlib import Path
from keras.preprocessing import image


# In[2]:


p=Path('images')

dirs=p.glob('*')

label_dict={'cat':1,'dog':2,'horse':3,'human':4}

labels=[]
image_data=[]

for folder_name in dirs:
    #print(folder_name)
    label_name=str(folder_name).split('\\')[-1][:-1]
    for img_path in folder_name.glob('*.jpg'):
        #print(img_path)
        
        # load image by a given path
        
        img=image.load_img(img_path,target_size=(32,32))
        # convert this loaded image into array
        img_array=image.img_to_array(img)
        
        image_data.append(img_array)
        labels.append(label_dict[label_name])


# In[3]:


# convert dataset into numpy array
image_data=np.array(image_data,dtype='float32')/255.0


# In[4]:


image_data.shape


# In[5]:


labels=np.array(labels)


# In[6]:


labels.shape


# In[7]:


def draw_image(data):
    import matplotlib.pyplot as plt
    plt.imshow(data)
    return


# In[8]:


#randomise our datasets
labels=labels.reshape((labels.shape[0],1))
zipped=np.concatenate((image_data.reshape(808,-1),labels),axis=1)
np.random.shuffle(zipped)
image_data=zipped[:,:-1].reshape(808,32,32,3)
labels=zipped[:,-1].reshape((808,))


# In[9]:


draw_image(image_data[0])


# In[10]:


class SVM:
    def __init__(self,C=1.00):
        self.C=C
        self.W=0
        self.b=0
        
    def hingeloss(self,W,b,X,y):
        loss=0.0
        loss+=0.5*np.dot(W,W.T)
        
        m=X.shape[0]
        for i in range(m):
            ti=y[i]*(np.dot(W,X[i].T)+b)
            loss+=self.C*max(0,1-ti)
        return loss[0][0]
    
    def fit(self,X,y,batch_size=100,learning_rate=0.001,maxitr=50):
        
        no_of_features=X.shape[1]
        no_of_samples=X.shape[0]
        
        n=learning_rate
        c=self.C
        
        #initialize the model parameters
        W=np.zeros((1,no_of_features))
        b=0
        
        losses=[]
        
        for i in range(maxitr):
            losses.append(self.hingeloss(W,b,X,y))
            ids=np.arange(no_of_samples)
            np.random.shuffle(ids)
            for batch_start in range(0,no_of_samples,batch_size):
                gradw=0.0
                gradb=0.0
                
                for j in range(batch_start,batch_start+batch_size):
                    if j<no_of_samples:
                        idx=ids[j]
                        ti=y[idx]*(np.dot(W,X[idx].T)+b)
                        if ti>=1:
                            gradw=0
                            gradb=0
                        else:
                            gradw+=c*y[idx]*X[idx]
                            gradb+=c*y[idx]
                W=W- n*W + n*gradw
                b=b - n*gradb
        self.W=W
        self.b=b
        return W,b,losses                


# ### Convert data into one for one classification :

# In[11]:


image_data=image_data.reshape(image_data.shape[0],-1)
print(image_data.shape)
print(labels.shape)


# ### Seperating data class-wise:

# In[12]:


def classwise_data(X,y):
    # creates a dictionary which stores data of different classes
    data={}
    
    n_classes=len(np.unique(y))
    for i in range(1,n_classes+1):
        # initialise every class data by empty list
        data[i]=[]
    
    for i in range(X.shape[0]):
        data[y[i]].append(X[i])
        
    for i in range(1,1+n_classes):
        data[i]=np.array(data[i])
        
    return data


# In[13]:


data=classwise_data(image_data,labels)


# In[14]:


data[1].shape


# In[15]:


def getPairsForSVM(d1,d2):
    l1=d1.shape[0]
    l2=d2.shape[0]
    
    n_samples=l1+l2
    n_features=d1.shape[1]
    
    data_pairs=np.zeros((n_samples,n_features))
    labels=np.zeros((n_samples,))
    
    data_pairs[:l1,:]=d1
    data_pairs[l1:,:]=d2
    
    labels[:l1]=-1
    labels[l1:]=1
    
    return data_pairs,labels


# In[16]:


mySVM=SVM()


# ### Train SVM for every pair of classes :

# In[45]:


def trainSVM(X,y):
    svm_classifiers={}
    
    n_classes=len(np.unique(y))
    for i in range(1,1+n_classes):
        svm_classifiers[i]={}
        
    for i in range(1,1+n_classes):
        for j in range(i+1,1+n_classes):
            x_pairs,y_pairs=getPairsForSVM(data[i],data[j])
            # reduce the learning rate and increase the iterations to increase the accuracy
            W,b,losses=mySVM.fit(x_pairs,y_pairs,batch_size=100,learning_rate=0.00001,maxitr=100)
            svm_classifiers[i][j]=(W,b)
            
    return svm_classifiers


# In[46]:


svm_classifiers=trainSVM(image_data,labels)


# ### Predict the training data :

# In[47]:


def decimal_prediction(x,W,b):
    p=np.dot(W,x.T) + b
    if p>0:
        return 1
    else:
        return -1


def predict(train_data):
    n_classes=len(np.unique(labels))
    count=np.zeros((n_classes+1,))
        
    for i in range(1,n_classes+1):
        for j in range(i+1,n_classes+1):
            W,b=svm_classifiers[i][j]
            z=decimal_prediction(train_data,W,b)
            if z==1:
                count[j]+=1
            else:
                count[i]+=1
    idx=np.argmax(count)
    return count[idx]


# In[48]:


predict(image_data[1])


# In[49]:


def accuracy(X,y):
    count=0.0
    for i in range(X.shape[0]):
        pred=predict(X[i])
        if pred==y[i]:
            count+=1
    return count/X.shape[0]


# In[50]:


accuracy(image_data,labels)


# # Using Sk-learn :

# In[53]:


from sklearn import svm
svc=svm.SVC(kernel='rbf',C=1.00)
svc.fit(image_data,labels)
svc.score(image_data,labels)


# In[ ]:




