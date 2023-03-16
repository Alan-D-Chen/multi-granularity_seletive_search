# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: 有用的代码.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,四月 09
# ---

# coding: utf-8
import cv2
import numpy as np

# 点集
points = []

# 窗口
window_name = 'DEMO'
window = cv2.namedWindow(window_name)

# 底图
img = np.zeros((300,300,3), dtype=np.uint8)

# 鼠标回调
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Click at (%d,%d)' % (x,y))
        points.append((x,y))
        cv2.circle(img, (x,y), 3, (255,255,255), -1)

# 程序入口
def main():
    global points, img
    # 设置回调
    cv2.setMouseCallback(window_name, on_mouse)

    # 画图
    while True:
        cv2.imshow(window_name, img)
        k = cv2.waitKey(1)
        if k == ord('q'): # 退出
            print ('EXIT')
            return
        elif k == ord('s'): # 绘制矩形
            (x,y,w,h) = cv2.boundingRect(np.array(points))
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        elif k == ord('c'): # 清空
            img[...] = 0
            points = []


if __name__ == '__main__':
    main()
