# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:09:15 2018

@author: Rob_hawk
"""

import cv2
import numpy as np
import time

cap=cv2.VideoCapture(0)
count = 0
while(True):
    ret, frame = cap.read()
    frame = frame 
    #kernel = np.ones((3,3),np.uint8)
    roi = frame
   
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    path="C:\\Users\\User\\Desktop\\dragon\\dragon"
    count+=1
    file_name_path= path+str(count)+".jpg"
    cv2.imwrite(file_name_path,roi)
    
    cv2.imshow("frame",frame)
    cv2.imshow("roi",roi)
    if(cv2.waitKey(1) == 13 or count==500):
        break;
    time.sleep(.005)
        
cap.release()
cv2.destroyAllWindows()
    