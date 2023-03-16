# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: 可能有用的代码2.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,四月 09
# ---
# coding: utf-8
import cv2
import numpy as np

# 程序入口
def main():
    img = cv2.imread('E:/dataset_2/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000076.jpg')
    gray = img[..., 0]

    # 寻找点集
    _, contours, _ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 画点
    for cnt in contours:
        for point in cnt:
            point = (point[0][0], point[0][1])
            cv2.circle(img, point, 1, (255,0,0), -1)

    # 画图
    cv2.imshow('DEMO', img)
    cv2.imwrite('contours.jpg', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
