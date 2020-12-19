#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import MaxPooling2D,Convolution2D,Flatten,Dense,Dropout
from keras import models 


# In[2]:


model=models.Sequential()
model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(4,activation='softmax'))
model.summary()


# In[3]:


model.compile('adam','categorical_crossentropy',metrics=['accuracy'])


# In[19]:


import os,shutil

# if we do not have a validation folder then create it
if not os.path.isdir('val_images'):
    os.mkdir('val_images')
    
classes=['cats','dogs','horses','humans']

# we have to make folders of images inside validation folder
for c in classes:
    p=os.path.join('val_images',c)
    if not p:
        os.mkdir(p)


# In[26]:


split=0.9
for folder in os.listdir('images'):
    path='images/'+folder
    img=os.listdir(path)
    
    split_len=int(len(img)*split)
    img_to_move=img[split_len:]
    
    for img_f in img_to_move:
        src=os.path.join(path,img_f)
        dest=os.path.join('val_images/'+folder,img_f)
        shutil.move(src,dest)


# In[4]:


# for small datasets we can use model.fit() method as it is fit inside a memory
from keras.preprocessing.image import ImageDataGenerator

train_gen=ImageDataGenerator(rescale=1/255.0,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.3,horizontal_flip=True)
val_gen=ImageDataGenerator(rescale=1/255.0)


# In[5]:


# flow_from_directory do load images, read images , convert it into array by itself
train_generator=train_gen.flow_from_directory('images',target_size=(150,150),batch_size=32)
val_generator=val_gen.flow_from_directory('val_images',target_size=(150,150),batch_size=32)


# In[6]:


train_generator.labels


# In[7]:


# iterate over all datasets
for x,y in train_generator:
    #x,y=train_generator.next()
    print(x.shape) # where x contains array of 32 images
    print(y.shape) # where y contain one_hot notation of 32 images
    break


# In[8]:


# now we have to fit our model
hist=model.fit_generator(train_generator,steps_per_epoch=7,epochs=40,validation_data=val_generator,validation_steps=4)


# In[9]:


import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.figure(figsize=(7,7))
plt.plot(hist.history['accuracy'],label='accuracy')
plt.plot(hist.history['val_accuracy'],label='validation accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()
plt.plot(hist.history['loss'],label='loss')
plt.plot(hist.history['val_loss'],label='validation loss')
plt.title('Loss')
plt.legend()
plt.show()


# In[37]:





# In[ ]:




