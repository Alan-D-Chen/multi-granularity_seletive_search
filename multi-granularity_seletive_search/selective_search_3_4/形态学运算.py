# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: 形态学运算.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,四月 09
# ---

import cv2
import numpy as np
import skimage
import skimage.morphology
import skimage.data
import skimage.io
import skimage.exposure
import skimage.color
######### Skimage图像处理教程5）形态学运算  ##########

'''
# 移除小的区域
skimage.morphology.remove_small_objects(ar, min_size=64, connectivity=1, in_place=False)
#移除小的孔洞
skimage.morphology.remove_small_holes(ar, area_threshold=64, connectivity=1, in_place=False, min_size=None)

# 图像局部重构
# 很像ppt中的删除图像背景的功能,seed为种子点，mask为图像。
skimage.morphology.reconstruction(seed, mask, method='dilation', selem=None, offset=None)


# 骨架提取
skimage.morphology.skeletonize(image)
skimage.morphology.skeletonize_3d(img)
skimage.morphology.thin(image, max_iter=None) # 没有iter参数也是给出骨架
skimage.morphology.medial_axis(image, mask=None, return_distance=False)
'''
###################################

img = skimage.data.binary_blobs(100)
#img = skimage.io.imread('E:/dataset_2/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000006.jpg')
img = skimage.color.rgb2gray(img)
skimage.io.imshow(img)
skimage.io.show()

img_ro = skimage.morphology.remove_small_objects(img, 128)
skimage.io.imshow(img_ro)
skimage.io.show()

mask = np.zeros(shape = (100,100))
seed[20,20] = 1
img_recon = skimage.morphology.reconstruction(seed,img )
skimage.io.imshow(img_recon)
skimage.io.show()

img_sk = skimage.morphology.skeletonize(img)
skimage.io.imshow(img_sk)
skimage.io.show()

img_thin = skimage.morphology.thin(img, max_iter=3)
skimage.io.imshow(img_thin)
skimage.io.show()

img_med = skimage.morphology.medial_axis(img)
skimage.io.imshow(img_med)
skimage.io.show()
