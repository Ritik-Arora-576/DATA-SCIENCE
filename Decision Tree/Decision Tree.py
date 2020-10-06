#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('titanic.csv')
df


# In[3]:


# eliminating the irrelevant informations

columns_to_drop=['Name','Ticket','Cabin','Embarked','PassengerId']


# In[4]:


df_cleaned=df.drop(columns=columns_to_drop,axis=1)


# In[5]:


df_cleaned


# In[6]:


# convert the sex column into numeric data because our model accept only numeric data
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_cleaned['Sex']=le.fit_transform(df_cleaned['Sex'])


# In[7]:


df_cleaned


# In[8]:


df_cleaned.info()


# In[9]:


# fill the void values
df_cleaned=df_cleaned.fillna(df_cleaned['Age'].mean())


# In[10]:


df_cleaned.info()


# In[11]:


# split data into input dataframe and output dataframe
input_data=['Pclass','Sex','Age','SibSp','Parch','Fare']
output_data=['Survived']


# In[12]:


X=df_cleaned[input_data]
y=df_cleaned[output_data]


# # Entropy :

# In[13]:


def entropy(col):
    uni=np.unique(col)
    total_samples=col.shape[0]
    n_classes=len(uni)
    
    ent=0.0
    for i in range(n_classes):
        favorable=np.sum(uni[i]==col)
        p=favorable/total_samples
        ent+=p*np.log2(p)
        
    return -1*ent


# In[14]:


# divide the data :

def divide_data(x_data,fkey,fval):
    x_left=pd.DataFrame([],columns=x_data.columns)
    x_right=pd.DataFrame([],columns=x_data.columns)
    
    for i in range(x_data.shape[0]):
        value=x_data[fkey].iloc[i]
        
        if value>fval:
            x_right=x_right.append(x_data.iloc[i])
        else:
            x_left=x_left.append(x_data.iloc[i])
            
    return x_left,x_right


# In[15]:


# find the information gain :

def information_gain(x_data,fkey,fval):
    
    total=x_data.shape[0]
    x_left,x_right=divide_data(x_data,fkey,fval)
    
    l=(x_left.shape[0])/total
    r=(x_right.shape[0])/total
    
    # in order to manage zero division error
    if l==0 or r==0:
        return -10000000  # Minimum information gain
    
    inf_gain=entropy(x_data['Survived'])-(l*entropy(x_left['Survived'])+r*entropy(x_right['Survived']))
    
    return inf_gain


# In[16]:


for col in X.columns:
    print(col)
    print(information_gain(df_cleaned,col,df_cleaned[col].mean()))


# ### Making Decision-Tree class :

# In[17]:


class DecisionTree:
    
    # making constructor
    def __init__(self,depth=0,max_depth=5):
        self.left=None
        self.right=None
        self.fval=None
        self.fkey=None
        self.depth=depth
        self.max_depth=max_depth
        self.target=None
        
    def train(self,x_train):
        features=['Pclass','Sex','Age','SibSp','Parch','Fare']
        info_gain=[]
        
        for i in features:
            i_gain=information_gain(x_train,i,x_train[i].mean())
            info_gain.append(i_gain)
            
        self.fkey=features[np.argmax(info_gain)]
        self.fval=x_train[self.fkey].mean()
        #print('Making tree ',self.fkey)
        #splitting the data
        x_left,x_right=divide_data(x_train,self.fkey,self.fval)
        x_left=x_left.reset_index(drop=True)
        x_right=x_right.reset_index(drop=True)
        
        # stop when we reach a leave node
        if x_left.shape[0]==0 or x_right.shape[0]==0:
            if x_train['Survived'].mean()>=0.5:
                self.target='Survive'   
            else:
                self.target='Dead'   
            return
        
        # stop when our depth is more than max_depth
        if(self.depth>=self.max_depth):
            if x_train['Survived'].mean()>=0.5:
                self.target='Survive'   
            else:
                self.target='Dead'   
            return
        
        # recursion call
        self.left=DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.left.train(x_left)
        
        self.right=DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.right.train(x_right)
        
        # You can set target at every node (i.e if a node is nighther a leaf node or at maximum depth)
        if x_train['Survived'].mean()>=0.5:
                self.target='Survive'   
        else:
                self.target='Dead'   
        return
    
    def predict(self,test):
        
        if test[self.fkey]>self.fval:
            
            #then it is go to right side
            if self.right is None:
                return self.target
            else:
                return self.right.predict(test)
        
        else:
            #then go to left side
            if self.left is None:
                return self.target
            else:
                return self.left.predict(test)
            


# In[29]:


split=int(0.8*df_cleaned.shape[0])
d=DecisionTree()
d.train(df_cleaned.iloc[:split,:])


# In[31]:


x_test=X.iloc[split:,:]
pred=[]
for i in range(x_test.shape[0]):
    pred.append(d.predict(x_test.iloc[i,:]))


# In[33]:


pred=le.fit_transform(pred)


# In[40]:


pred


# In[48]:


np.sum(y.iloc[split:,:].values.reshape((-1,))==pred)/len(pred)


# # Decision Tree using sk-learn :

# In[50]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy',max_depth=5)


# In[51]:


dtc.fit(X.iloc[:split,:],y.iloc[:split,:])


# In[53]:


pred=dtc.predict(X.iloc[split:,:])


# In[56]:


np.sum(y.iloc[split:,:].values.reshape((-1,))==pred)/len(pred)


# In[ ]:




