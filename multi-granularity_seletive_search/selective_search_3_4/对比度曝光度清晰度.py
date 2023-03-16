# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: 对比度曝光度清晰度.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,四月 11
# ---
'''
######################################
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
image = img_as_float(data.moon())
gam1= exposure.adjust_gamma(image, 2)   #调暗
gam2= exposure.adjust_gamma(image, 0.5)  #调亮
plt.figure('adjust_gamma',figsize=(8,8))

plt.subplot(131)
plt.title('origin image')
plt.imshow(image,plt.cm.gray)
plt.axis('off')

plt.subplot(132)
plt.title('gamma=2')
plt.imshow(gam1,plt.cm.gray)
plt.axis('off')

plt.subplot(133)
plt.title('gamma=0.5')
plt.imshow(gam2,plt.cm.gray)
plt.axis('off')

plt.show()
'''

'''
################################################
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
image = img_as_float(data.moon())
gam1= exposure.adjust_log(image)   #对数调整
plt.figure('adjust_gamma',figsize=(8,8))

plt.subplot(121)
plt.title('origin image')
plt.imshow(image,plt.cm.gray)
plt.axis('off')

plt.subplot(122)
plt.title('log')
plt.imshow(gam1,plt.cm.gray)
plt.axis('off')

plt.show()
'''

'''
判断图像对比度是否偏低
函数：is_low_contrast(img)
返回一个bool型值
'''

from skimage import data, exposure
image =data.moon()
result=exposure.is_low_contrast(image)
print("is_low_contrast(image)??",result)

