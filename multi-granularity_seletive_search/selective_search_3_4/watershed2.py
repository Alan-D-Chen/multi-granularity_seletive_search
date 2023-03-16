# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: watershed2.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,五月 01
# ---
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology,color,data,filters
import cv2
from PIL import Image

image = cv2.imread('E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2008_003726.jpg')
#image = Image.open('E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2011_002371.jpg')
image =color.rgb2gray(image)
denoised = filters.rank.median(image, morphology.disk(2)) #过滤噪声

#将梯度值低于10的作为开始标记点
markers = filters.rank.gradient(denoised, morphology.disk(5)) <10
markers = ndi.label(markers)[0]

gradient = filters.rank.gradient(denoised, morphology.disk(2)) #计算梯度
labels = morphology.watershed(gradient, markers, mask=image) #基于梯度的分水岭算法

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(image)#, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title("Original")
ax1.imshow(gradient)#, cmap=plt.cm.Spectral, interpolation='nearest')
ax1.set_title("Gradient")
ax2.imshow(markers)#, cmap=plt.cm.Spectral, interpolation='nearest')
ax2.set_title("Markers")
ax3.imshow(labels)#, cmap=plt.cm.Spectral, interpolation='nearest')
ax3.set_title("Segmented")

for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()