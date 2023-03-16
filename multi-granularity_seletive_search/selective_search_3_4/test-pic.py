# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: test-pic.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,五月 15
# ---
from __future__ import division
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy
import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology,color,data,filters
import matplotlib
import matplotlib.pyplot as plt

image = cv2.imread('E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000783.jpg')

# hsv = skimage.color.rgb2hsv(image)
# lab = skimage.color.rgb2lab(image)
# luv = skimage.color.rgb2luv(image)
# yuv = skimage.color.rgb2yuv(image)
#
# image2 = cv2.Laplacian(image, cv2.CV_64F)
# binf = cv2.inRange(image, np.array([100, 43, 46]), np.array([124, 255, 255]))
# binf2 = cv2.inRange(image, np.array([90, 43, 46]), np.array([130, 255, 255]))
#imgs3 = Image.fromarray(rgb.astype('uint8')).convert('RGB')
#imgs = ImageEnhance.Contrast(image).enhance(2.0)
#imgs2 = ImageEnhance.Sharpness(imgs).enhance(2.0)
#gray = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2GRAY)
# gray = np.asarray(gray)

# cv2.imshow("image",image)
# cv2.imshow("hsv",hsv)
# cv2.imshow("lab",lab)
# cv2.imshow("luv",luv)
# cv2.imshow("yuv",yuv)
# #v2.imshow("gray",gray)
# cv2.imshow("binf",binf)
# cv2.imshow("binf2",binf2)
# cv2.imshow("image2",image2)
# # cv2.imshow("imgs2",imgs2)
# # cv2.imshow("imgs",imgs)
# cv2.waitKey(0)

# plt.imshow(image ,cmap="gray")
# plt.axis('off')
# plt.show()

# im=plt.imread('E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000783.jpg', pyplot.cm.gray)
# plt.imshow(im)

imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #彩色转灰度
cv2.imshow("imgray", imgray)
cv2.waitKey(0)