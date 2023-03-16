# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: 跟踪物体.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,四月 09
# ---

import cv2
import numpy as np
cap=cv2.VideoCapture(0)

while(1):
    ret,frame=cap.read()#读取视频帧
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#HSV空间

    lower_blue=np.array([50,50,80])#设定蓝色的阈值
    upper_blue=np.array([255,255,180])

    mask=cv2.inRange(frame,lower_blue,upper_blue)#设定取值范围
    res=cv2.bitwise_and(frame,frame,mask=mask)#对原图像处理

    cv2.imshow('frame-original:',frame)
    cv2.imshow('mask-Binary figure:',mask)
    cv2.imshow('res:',res)
    k=cv2.waitKey(5)&0xFF
    if k==27:
        break
cv2.destroyAllWindows()
