# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: 清晰度.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,四月 13
# ---

import cv2
import numpy as np

image = cv2.imread('E:/dataset_2/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000001.jpg')

def getImageVar(imgPath):
    imageVar0 = cv2.Laplacian(image, cv2.CV_64F).var()

    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()

    image2 = cv2.Laplacian(image, cv2.CV_64F)
    imageVar2 = cv2.Laplacian(image2, cv2.CV_64F).var()

    image3 = cv2.Laplacian(img2gray, cv2.CV_64F)
    imageVar3 = cv2.Laplacian(image3, cv2.CV_64F).var()

    return img2gray,image2,image3,imageVar0,imageVar,imageVar2,imageVar3


if __name__ == '__main__':
    image1,image2,image3,a,b,c,d = getImageVar(image)
    cv2.imshow('origin image:',image)
    print('origin image Laplacian Var:',a)
    cv2.imshow('img2gray:',image1)
    print("image2gray Laplacian Var:",b)
    cv2.imshow('origin Laplacian image2:',image2)
    print("origin Laplacian image Laplacian Var:",c)
    cv2.imshow('img2gray Laplacian image:',image3)
    print('img2gray Laplacian image Lapacian Var',d)
    cv2.waitKey(0)