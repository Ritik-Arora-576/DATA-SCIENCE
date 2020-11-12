#!/usr/bin/env python
# coding: utf-8

# ### PART-1 (IMPLEMENTATION)

# In[82]:


import numpy as np
input_size=2 # number of features we want
layers=[4,3] # number of layers in 1st and 2nd layer
output_size=2


# In[83]:


def softmax(a):
    e=np.exp(a)
    ans=e/np.sum(e,axis=1,keepdims=True)
    return ans


# In[84]:


class NeuralNetwork:
    
    def __init__(self,input_size,layers,output_size):
        np.random.seed(0) # used to generate same random data everytime
        
        model={} # dictionary contain wieghts and biased term for each layer
        
        # First layer
        model['W1']=np.random.randn(input_size,layers[0])
        model['b1']=np.random.randn(1,layers[0])
        
        # Second layer
        model['W2']=np.random.randn(layers[0],layers[1])
        model['b2']=np.random.randn(1,layers[1])
        
        # Third / Output layer
        model['W3']=np.random.randn(layers[1],output_size)
        model['b3']=np.random.randn(1,output_size)
        
        self.model=model
        self.input_size=input_size
        self.layers=layers
        self.output_size=output_size
        
    def forward(self,x):
        W1,W2,W3=self.model['W1'],self.model['W2'],self.model['W3']
        b1,b2,b3=self.model['b1'],self.model['b2'],self.model['b3']
        
        z1=np.dot(x,W1) + b1 # used to reduce n-dimensional data to 4 dimensional data
        a1=np.tanh(z1)
        
        z2=np.dot(a1,W2) + b2  # used to reduce 4-dimensional data to 3 dimensional data
        a2=np.tanh(z2)
        
        z3=np.dot(a2,W3) + b3  # used to reduce 3-dimensional data to 2 dimensional data
        a3=np.tanh(z3)
        
        y_=softmax(a3) # activation function
        
        self.activation_output=(a1,a2,y_)
        return y_
    
    def backward(self,x,y,learning_rate=0.001):
        W1,W2,W3=self.model['W1'],self.model['W2'],self.model['W3']
        b1,b2,b3=self.model['b1'],self.model['b2'],self.model['b3']
        
        a1,a2,y_=self.activation_output
        
        m=x.shape[0]
        
        delta3=y_-y # delta of output layer
        
        dw3=np.dot(a2.T,delta3) # change in Weights of output layer
        db3=np.sum(delta3,axis=0)/float(m)
        
        delta2=(1-np.square(a2))*np.dot(delta3,W3.T)
        
        dw2=np.dot(a1.T,delta2)
        db2=np.sum(delta2,axis=0)/float(m)
        
        delta1=(1-np.square(a1))*np.dot(delta2,W2.T)
        
        dw1=np.dot(x.T,delta1)
        db1=np.sum(delta1,axis=0)/float(m)
        
        n=learning_rate
        # update the value of weights and biased term
        
        self.model['W1']=self.model['W1']- (n*dw1)
        self.model['b1']=self.model['b1']- (n*db1)
        
        self.model['W2']=self.model['W2']- (n*dw2)
        self.model['b2']=self.model['b2']- (n*db2)
        
        self.model['W3']=self.model['W3']- (n*dw3)
        self.model['b3']=self.model['b3']- (n*db3)
        
    def prediction(self,x):
        pred=self.forward(x)
        return np.argmax(pred,axis=1)
    
    def summary(self):
        W1,W2,W3=self.model['W1'],self.model['W2'],self.model['W3']
        a1,a2,y_=self.activation_output
        
        print('W1 ',W1.shape)
        print('a1 ',a1.shape)
        
        print('W2 ',W2.shape)
        print('a2 ',a2.shape)
        
        print('W3 ',W3.shape)
        print('y_ ',y_.shape)


# In[85]:


def loss(y_oht,p):
    return -np.mean(y_oht*np.log(p))


# In[86]:


def one_hot(y,depth):
    m=y.shape[0]
    y_oht=np.zeros((m,depth))
    
    for i in range(m):
        y_oht[i,y[i]]=1
        
    return y_oht


# In[87]:


from sklearn.datasets import make_circles


# In[88]:


X,y = make_circles(n_samples=500, shuffle=True, noise=0.2, random_state=1, factor=0.2)


# In[89]:


import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


# In[90]:


y


# In[91]:


y_oht=one_hot(y,2)


# ### Training our model :

# In[92]:


model=NeuralNetwork(2,[10,5],2)

def train(X,y,model,max_itr=500,learning_rate=0.001):
    losses=[]
    classes=2
    y_oht=one_hot(y,classes) # actual classes
    
    for i in range(max_itr):
        y_=model.forward(X) # prediction of each classes
        l=loss(y_oht,y_)
        model.backward(X,y_oht,learning_rate)
        losses.append(l)
        
    return losses


# In[93]:


losses=train(X,y,model)


# In[94]:


plt.plot(losses)


# In[95]:


pred=model.prediction(X)


# ### Accuracy :

# In[96]:


np.sum(pred==y)/y.shape[0]


# In[97]:


X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]
             ])

Y = np.array([0,1,1,0])


# In[98]:


losses=train(X,Y,model)


# In[99]:


plt.plot(losses)


# In[101]:


pred=model.prediction(X)


# In[103]:


np.sum(Y==pred)/Y.shape[0]


# In[105]:


from sklearn.datasets import make_moons
X,Y = make_moons(n_samples=500,noise=0.2,random_state=1)


# In[106]:


losses=train(X,Y,model)


# In[107]:


plt.plot(losses)
plt.show()


# In[108]:


pred=model.prediction(X)
np.sum(Y==pred)/Y.shape[0]


# In[ ]:




