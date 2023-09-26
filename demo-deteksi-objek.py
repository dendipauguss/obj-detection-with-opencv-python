#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import matplotlib.pyplot as plt


# In[2]:


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'


# In[3]:


model = cv2.dnn_DetectionModel(frozen_model, config_file)


# In[4]:


classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


# In[5]:


print(classLabels)


# In[6]:


print(len(classLabels))


# In[7]:


model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127,5,127.5))
model.setInputSwapRB(True)


# In[8]:


img = cv2.imread('bmw-man.jpg')
plt.imshow(img)


# In[9]:


ClassIndex, confidece, bbox = model.detect(img, confThreshold = 0.5)


# In[10]:


print(ClassIndex)


# In[11]:


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img, boxes,(255,0,0),2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color=(0,255,0), thickness=3)


# In[12]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


# Video
cap = cv2.VideoCapture('lalu-lintas.mp4')

if not cap.isOpened():
    cap= cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Can't open the video")
    
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret,frame = cap.read()

    ClassIndex, confidece, bbox = model.detect(frame, confThreshold = 0.55)

    print(ClassIndex)

    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame, boxes,(255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color=(0,255,0), thickness=3)

    cv2.imshow('deteksi objek by DPS', frame)

    if cv2.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




