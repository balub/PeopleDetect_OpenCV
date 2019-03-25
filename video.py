# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:14:23 2019

@author: Balu
"""

import numpy as np
import cv2


cap = cv2.VideoCapture('video2.MP4')
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    human = human_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in human:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        
    cv2.imshow('img',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
