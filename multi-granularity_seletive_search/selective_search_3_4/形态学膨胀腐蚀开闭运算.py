# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: 形态学膨胀腐蚀开闭运算.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,四月 11
# ---
import skimage
import skimage.morphology
import skimage.data
import skimage.io
import skimage.exposure

#######  Skimage图像处理教程5）形态学运算  ##########
'''
# 膨胀
skimage.morphology.dilation(image, selem=None, out=None, shift_x=False, shift_y=False)

# 腐蚀
skimage.morphology.erosion(image, selem=None, out=None, shift_x=False, shift_y=False)

# 开运算
skimage.morphology.opening(image, selem=None, out=None)

# 闭运算
skimage.morphology.closing(image, selem=None, out=None)

# 白顶帽操作，原图减去开运算的结果
skimage.morphology.white_tophat(image, selem=None, out=None)

# 黑顶帽操作，闭运算的结果减去原图
skimage.morphology.black_tophat(image, selem=None, out=None)

##### 除此之外做形态学操作一定会用到各种形态学算子下面就列出一些常用的算子  #####

skimage.morphology.square(width, dtype=<class 'numpy.uint8'>)    #正方形
skimage.morphology.rectangle(width, height, dtype=<class 'numpy.uint8'>)    #长方形
skimage.morphology.diamond(radius, dtype=<class 'numpy.uint8'>)    #钻石形
skimage.morphology.disk(radius, dtype=<class 'numpy.uint8'>)    #圆形
skimage.morphology.cube(width, dtype=<class 'numpy.uint8'>)    #立方体
skimage.morphology.octahedron(radius, dtype=<class 'numpy.uint8'>)    #八面体
skimage.morphology.ball(radius, dtype=<class 'numpy.uint8'>)    #球体
skimage.morphology.octagon(m, n, dtype=<class 'numpy.uint8'>)    #八角形
skimage.morphology.star(a, dtype=<class 'numpy.uint8'>)    #星形

for example
'''
img = skimage.data.binary_blobs(100)
skimage.io.imshow(img)
skimage.io.show()

kernel = skimage.morphology.disk(3)
img_dialtion = skimage.morphology.dilation(img, kernel)
skimage.io.imshow(img_dialtion)
skimage.io.show()

img_erosion = skimage.morphology.erosion(img, kernel)
skimage.io.imshow(img_erosion)
skimage.io.show()

img_open = skimage.morphology.opening(img, kernel)
skimage.io.imshow(img_open)
skimage.io.show()

img_close = skimage.morphology.closing(img, kernel)
skimage.io.imshow(img_close)
skimage.io.show()

img_white = skimage.morphology.white_tophat(img, kernel)
skimage.io.imshow(img_white)
skimage.io.show()