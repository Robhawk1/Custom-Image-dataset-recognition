# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:01:29 2018

@author: Rob_hawk
"""

import cv2
import numpy as np
from keras.models import load_model
import os


path = 'C:\\Users\\User\\Desktop\\jutsu_api\\dataset'
classes = os.path.join(path, 'test')
labels = []

#img_path = os.path.join(classes, 'tiger\\tiger176.jpg')
for label in os.listdir(classes):
    labels.append(label)
    
model  = load_model('test_run.h5')

cap = cv2.VideoCapture(0)
"""
img = cv2.imread(img_path)

img_cpy = np.expand_dims(img, axis = 0)
pred = model.predict(img_cpy).argmax(axis = 1)

for _ in pred:
    label = _

typ = labels[label]

cv2.putText(img, typ, (10,10), cv2.FONT_HERSHEY_SIMPLEX,0.55, (0,255,0), 2 )

cv2.imshow('dragon', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
while True:
    
    ret, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)
    frame_cpy = cv2.resize(frame, (640,355))
    frame_cpy = np.expand_dims(frame_cpy, axis = 0)
    pred = model.predict(frame_cpy).argmax(axis = 1)
    
    for _ in pred:
        label = _
    typ = labels[label]
    cv2.putText(frame, typ, (10,10), cv2.FONT_HERSHEY_SIMPLEX,0.55, (0,255,0), 2 )
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 13:
        break;


cap.release()
cv2.destroyAllWindows()
