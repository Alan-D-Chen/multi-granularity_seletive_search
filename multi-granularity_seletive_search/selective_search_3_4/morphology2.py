# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: morphology2.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,五月 01
# ---
from skimage import morphology,data,color
import matplotlib.pyplot as plt
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy
import cv2

image = cv2.imread('E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000783.jpg')
image=color.rgb2gray(image)
image=1-image #反相
#实施骨架算法
skeleton =morphology.skeletonize(image)

#显示结果
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax1.imshow(image, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('original', fontsize=20)

ax2.imshow(skeleton, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()