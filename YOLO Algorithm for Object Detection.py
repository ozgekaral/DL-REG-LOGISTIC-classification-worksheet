# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:55:42 2023

@author: user202
"""

import cv2
import numpy as np
from yolo_model import YOLO

y=YOLO(0.6,0.5)
F='data/coco_classes.txt'

with open(F) as f:
    class_n=f.readlines()
all_classes=[c.strip() for c in class_n]

f='search.jpg'
path='Yeni klasör (2)/'+f
img=cv2.imread('path')

#Preprocess
img_=cv2.resize(img, (416,416))
img_=np.array(img_, dtype='float32')
img_=np.expand_dims(img_, axis=0)

#Predict
boxes, classes, scores=yolo.predict(img_, img.shape)
for box, score, cl in zip(boxes, scores, classes):
    x,y,w,h=box
    top=max(0, np.floor(x+0.5).astype(int))
    left=max(0, np.floor(y+0.5).astype(int))
    right=max(0, np.floor(x+w+0.5).astype(int))
    bottom=max(0, np.floor(y+h+0.5).astype(int))
#Rectangle
    cv2.rectangle(img, (top,left), (right,bottom), (255,0,0),2)
cv2.imshow('search', img)



