# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: 对比度色彩饱和度色调锐度.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,四月 13
# ---


import os
from PIL import Image
from PIL import ImageEnhance

#image = Image.open('E:/dataset_2/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000076.jpg')
image = Image.open('E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000783.jpg')
im02 = image
'''
#变亮     #亮度增强,增强因子为0.0将产生黑色图像；为1.0将保持原始图像。
enh_bri = ImageEnhance.Brightness(image)
brightness = 1.5
image_brightened1 = enh_bri.enhance(brightness)
image_brightened1.save(os.path.join(parent, '{}_bri1.jpg'.format(name)))
#变暗
enh_bri = ImageEnhance.Brightness(image)
brightness = 0.8
image_brightened2 = enh_bri.enhance(brightness)
image_brightened2.save(os.path.join(parent, '{}_bri2.jpg'.format(name)))
#色度,增强因子为1.0是原始图像
# 色度增强
enh_col = ImageEnhance.Color(image)
color = 1.5
image_colored1 = enh_col.enhance(color)
image_colored1.save(os.path.join(parent, '{}_col1.jpg'.format(name)))
# 色度减弱
enh_col = ImageEnhance.Color(image)
color = 0.8
image_colored1 = enh_col.enhance(color)
image_colored1.save(os.path.join(parent, '{}_col2.jpg'.format(name)))
#对比度，增强因子为1.0是原始图片
 #对比度增强
enh_con = ImageEnhance.Contrast(image)
contrast = 1.5
image_contrasted1 = enh_con.enhance(contrast)
image_contrasted1.save(os.path.join(parent, '{}_con1.jpg'.format(name)))
 #对比度减弱
enh_con = ImageEnhance.Contrast(image)
contrast = 0.8
image_contrasted2 = enh_con.enhance(contrast)
image_contrasted2.save(os.path.join(parent, '{}_con2.jpg'.format(name)))
# 锐度，增强因子为1.0是原始图片
 #锐度增强
enh_sha = ImageEnhance.Sharpness(image)
sharpness = 3.0
image_sharped1 = enh_sha.enhance(sharpness)
image_sharped1.save(os.path.join(parent, '{}_sha1.jpg'.format(name)))
# 锐度减弱
enh_sha = ImageEnhance.Sharpness(image)
sharpness = 0.8
image_sharped2 = enh_sha.enhance(sharpness)
image_sharped2.save(os.path.join(parent, '{}_sha2.jpg'.format(name)))
image
'''
######## 图片的色度增强 #########
# im_1 = ImageEnhance.Color(im02).enhance(0.1)
# im_5 = ImageEnhance.Color(im02).enhance(0.5)
# im_8 =ImageEnhance.Color(im02).enhance(0.8)
# im_20 = ImageEnhance.Color(im02).enhance(2.0)
# im_1.show()
# im_5.show()
# im_8.show()
# im_20.show()
####### 图片的亮度增加 #########
# im_2 = ImageEnhance.Brightness(im02).enhance(0.2)
# im_5 = ImageEnhance.Brightness(im02).enhance(0.5)
# im_8 =ImageEnhance.Brightness (im02).enhance(0.8)
# im_20 =ImageEnhance.Brightness (im02).enhance(2.0)
# im_2.show()
# im_5.show()
# im_8.show()
# im_20.show()
####### 对比度增强类用于调整图像的对比度。类似于调整彩色电视机的对比度 #########
im_1 = ImageEnhance.Contrast(im02).enhance(0.1)
im_5 = ImageEnhance.Contrast(im02).enhance(0.5)
im_8 = ImageEnhance.Contrast(im02).enhance(0.8)
im_20 = ImageEnhance.Contrast(im02).enhance(2.0)
im_30 = ImageEnhance.Contrast(im02).enhance(3.0)
im_1.show()
im_5.show()
im_8.show()
im_20.show()
im_30.show()
####### 锐度增强类用于调整图像的锐度 ##########
im_0 = ImageEnhance.Sharpness(im02).enhance(0.0)
im_20 =ImageEnhance.Sharpness(im02).enhance(2.0)
im_30 =ImageEnhance.Sharpness(im02).enhance(3.0)
im_40 =ImageEnhance.Sharpness(im02).enhance(4.0)
im_0.show()
im_20.show()
im_30.show()
im_40.show()


