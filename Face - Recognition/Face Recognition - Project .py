#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
image_data=cv2.imread('pikachu.jpg')# by default read into BGR format
image_data=cv2.cvtColor(image_data,cv2.COLOR_BGR2RGB) # to convert BGR image into RGB image
plt.imshow(image_data)
plt.axis('off')
plt.figure(figsize=(3,3))
plt.show()


# In[4]:


import cv2
image_data=cv2.imread('pikachu.jpg')
gray_image=cv2.imread('pikachu.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Pikachu image',image_data)
cv2.imshow('Pikachu b&w',gray_image)
cv2.waitKey(0) # use to hold output screen for infinite time
cv2.destroyAllWindows()


# # working with video stream :

# In[3]:


import cv2
import numpy as np

# Start a video camera
cap=cv2.VideoCapture(0) # Video Camera of ID 0

# Face detection - Using haarcascade
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip=0
face_data=[]
folder_name='Face_recognition/'

file_name=input('Enter the file name : ')

while True:
    bool_value,frame=cap.read() # captured image gives bool value and frame
    
    if bool_value==False: # if video camera is not capturing frame then skip the loop
        continue
        
    # Detect faces in a frame
    faces = face_cascade.detectMultiScale(frame,1.3,5) # gives a tubble of x,y,w,h
    # x,y=Cordinates of first phase
    # w,h=width and hieght of a face
    
    # Sort faces on the basis of frame size
    faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True)
    section_frame=frame
    for face in faces:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) # build rectangular frame around face
        
        # now crop the image from frame within a rectangular limits
        offset=10
        section_frame=frame[y-offset:y+offset+h,x-offset:x+offset+w] # frame having (y,x) by default cordinates
        section_frame=cv2.resize(section_frame,(100,100)) # Make a wndow of size 100x100
        
        skip+=1
        if skip%10==0:
            face_data.append(section_frame) # make a list of face_data after every 10th frame
        
        
    cv2.imshow('Video Frame',frame) # else show the image frame captured
    cv2.imshow('Section Frame',section_frame)
    
    key_pressed=cv2.waitKey(1) & 0xFF 
    if key_pressed==ord('q'): # if pressed key is q then break the loop and close the video stream
        break

# make a list into numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))

# save the face_data into numpy folder
np.save(folder_name+file_name+'.npy',face_data)

print('File succesfully saved at ',folder_name+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()


# In[1]:


import numpy as np
import os
import cv2

########## KNN ######################################################
def distance(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(x_values,y_values,query,k=5):
    total_points=y_values.shape[0]
    values=[]
    for i in range(total_points):
        values.append((distance(x_values[i],query),y_values[i]))
        
    values=np.array((sorted(values)[:k]))
    uni=np.unique(values[:,1],return_counts=True)
    index=uni[1].argmax()
    return uni[0][index]
######################################################################

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('haarcascade_Frontalface_alt.xml')
skip=0
face_data=[]
folder_name='Face_Recognition/'
labels=[]
class_id=0 # used to label the file
names={}

# Data Prepration
for file in os.listdir(folder_name): # iterate over each file in folder
    if file.endswith('.npy'):
        # creates a mapping
        names[class_id]=str(file[:-4])
        data_item=np.load(folder_name+file) # open and read the file
        face_data.append(data_item) # append the image information
        
        target=class_id*np.ones(data_item.shape[0])
        class_id+=1
        labels.append(target)
face_dataset=np.concatenate(face_data,axis=0)
label_dataset=np.concatenate(labels,axis=0).reshape(-1,1)
train_datasets=np.concatenate((face_dataset,label_dataset),axis=1)
#print(train_datasets.shape)

# Testing our data
while True:
    bool_value,frame=cap.read()
    if bool_value==False:
        continue
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    for face in faces:
        x,y,w,h=face
       
        offset=10
        section_frame=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        section_frame=cv2.resize(section_frame,(100,100))
        out=knn(train_datasets[:,:-1],train_datasets[:,-1],section_frame.flatten())
        predicted_name=names[int(out)]
        cv2.putText(frame,predicted_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('frame',frame)
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:




