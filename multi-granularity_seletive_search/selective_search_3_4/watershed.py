# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: watershed.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,五月 01
# ---

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology,feature
import cv2

#创建两个带有重叠圆的图像
x, y = np.indices((80, 80))
x1, y1, x2, y2 = 28, 28, 44, 52
r1, r2 = 16, 20
mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
image = np.logical_or(mask_circle1, mask_circle2)
#image = cv2.imread('E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg')

#现在我们用分水岭算法分离两个圆
distance = ndi.distance_transform_edt(image) #距离变换
local_maxi =feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=image)   #寻找峰值
markers = ndi.label(local_maxi)[0] #初始标记点
labels =morphology.watershed(-distance, markers, mask=image) #基于距离变换的分水岭算法

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title("Original")
ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
ax1.set_title("Distance")
ax2.imshow(markers, cmap=plt.cm.Spectral, interpolation='nearest')
ax2.set_title("Markers")
ax3.imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest')
ax3.set_title("Segmented")

for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()