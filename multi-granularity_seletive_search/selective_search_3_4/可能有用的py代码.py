# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: 可能有用的py代码.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,四月 08
# ---

import cv2
import numpy as np
import skimage
import skimage.morphology
import skimage.data
import skimage.io
import skimage.exposure

'''
函数名：rgb2gray_mean
功能：通过求通道平均值得到灰度图
输入：
img    输入的彩图
返回：
result    灰度图
'''

def rgb2gray_mean(img):
    ratio = 1.0 / 3
    # 转换类型
    int_img = img.astype(np.int32)
    result = ratio * (int_img[...,0]+int_img[...,1]+int_img[...,2])
    return result.astype(np.uint8)

'''
函数名：心理学模型转换灰度
输入：
img    输入的彩图
返回：
result    灰度图
'''
def rgb2gray_mental(img):
    # 转换类型
    int_img = img.astype(np.int32)
    result = (int_img[...,2]*299 + int_img[...,1]*587 + int_img[...,0]*114 + 500) / 1000
    return result.astype(np.uint8)

def main():

    # 读取lena图
    # img = cv2.imread('E:/dataset_2/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000076.jpg')

    # 读取lena图
    color = cv2.imread('E:/dataset_2/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000076.jpg')
    # 打印color的维度
    print('color.shape:', color.shape)
    # 打印color的一个像素值
    print('color[0,0]:', color[0, 0])
    # 打印color的一个像素的一个通道值
    print('color[0,0,0]:', color[0, 0, 0])

    #####################################  转灰度/HSV/二值图/滤波/切换色彩空间~########

    ######### 色彩空间 #######
    gray = rgb2gray_mean(color)
    gray2 = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray3 = rgb2gray_mental(color)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 43, 46]), np.array([124, 255, 255]))
    #mask2 = cv2.inRange(color, np.array([100, 43, 46]), np.array([124, 255, 255]))
    ###############################################################################
    mask2 = cv2.inRange(color, np.array([90, 43, 46]), np.array([130, 255, 255]))
    #### ---------------- ###########
    #mask9 = cv2.cvtColor(color, cv2.COLOR_BGR2BGR555)
    #mask10 = cv2.cvtColor(color, cv2.COLOR_BGR2BGR565)
    mask11 = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
    mask12 = cv2.cvtColor(color, cv2.COLOR_BGR2HLS)
    mask13 = cv2.cvtColor(color, cv2.COLOR_BGR2HLS_FULL)
    mask14 = cv2.cvtColor(color, cv2.COLOR_BGR2HSV_FULL)
    mask15 = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    mask16 = cv2.cvtColor(color, cv2.COLOR_BGR2LUV)
    mask17 = cv2.cvtColor(color, cv2.COLOR_BGR2Lab)
    mask18 = cv2.cvtColor(color, cv2.COLOR_BGR2Luv)
    mask19 = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    mask20 = cv2.cvtColor(color, cv2.COLOR_BGR2RGBA)
    mask21 = cv2.cvtColor(color, cv2.COLOR_BGR2XYZ)
    mask22 = cv2.cvtColor(color, cv2.COLOR_BGR2YCR_CB)
    mask23 = cv2.cvtColor(color, cv2.COLOR_BGR2YCrCb)
    mask24 = cv2.cvtColor(color, cv2.COLOR_BGR2YUV)
    #mask25 = cv2.cvtColor(color, cv2.COLOR_BGR2YUV_I420)
    #mask26 = cv2.cvtColor(color, cv2.COLOR_BGR2YUV_IYUV)
    #mask27 = cv2.cvtColor(color, cv2.COLOR_BGR2YUV_YV12)

    ###腐蚀与膨胀，开运算与闭运算####
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
    # 腐蚀
    mask3 = cv2.erode(mask2, kernel)
    # 膨胀
    mask4 = cv2.dilate(mask3, kernel)
    ###
    mask42 = cv2.erode(cv2.dilate(mask2, kernel),kernel)
    ##### 滤波处理 ####### 模糊处理 #####
    ## 中值滤波 ##
    mask5 = cv2.medianBlur(color, 9)
    ## 均值滤波 ##
    mask6 = cv2.blur(color,(3,3))
    ## 方框滤波 ##
    mask7 = cv2.boxFilter(color, -1, (3, 3), normalize=True)
    ## 高斯滤波
    mask8 = cv2.GaussianBlur(color,(3, 3), 1)

    ##################################################

    #################### 显示 #####################
    cv2.imshow('color', color)
    cv2.imshow('gray', gray)
    cv2.imshow('gray2', gray2)
    cv2.imshow('gray3', gray3)
    cv2.imshow('hsv', hsv)
    cv2.imshow('mask', mask)
    cv2.imshow('mask2', mask2)
    cv2.imshow('mask3 erode:', mask3)
    cv2.imshow('mask4 color->erode->dilate:', mask4)
    cv2.imshow('mask4 color->dilate->erode:', mask4)
    cv2.imshow('medianblur:', mask5)
    cv2.imshow('blur:', mask6)
    cv2.imshow('boxFilter:', mask7)
    cv2.imshow('GuassianBlur:', mask8)
    ################################################
    #cv2.imshow('COLOR_BGR2BGR555:', mask9)
    #cv2.imshow('COLOR_BGR2BGR565:', mask10)
    cv2.imshow('COLOR_BGR2BGRA:', mask11)
    cv2.imshow('COLOR_BGR2HLS:', mask12)
    cv2.imshow('COLOR_BGR2HLS_FULL:', mask13)
    cv2.imshow('COLOR_BGR2HSV_FULL:', mask14)
    cv2.imshow('COLOR_BGR2LAB:', mask15)
    cv2.imshow('COLOR_BGR2LUV:', mask16)
    cv2.imshow('COLOR_BGR2Lab:', mask17)
    cv2.imshow('COLOR_BGR2Luv:', mask18)
    cv2.imshow('COLOR_BGR2RGB:', mask19)
    cv2.imshow('COLOR_BGR2RGBA:', mask20)
    cv2.imshow('COLOR_BGR2XYZ:', mask21)
    cv2.imshow('COLOR_BGR2YCR_CB:', mask22)
    cv2.imshow('COLOR_BGR2YCrCb:', mask23)
    cv2.imshow('COLOR_BGR2YUV:', mask24)
    #cv2.imshow('COLOR_BGR2YUV_I420	:', mask25)
    #cv2.imshow('COLOR_BGR2YUV_IYUV	:', mask26)
    #cv2.imshow('COLOR_BGR2YUV_YV12	:', mask27)
    cv2.waitKey(0)

if __name__ == '__main__':

    main()
