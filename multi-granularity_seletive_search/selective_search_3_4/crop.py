# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: crop.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,五月 08
# ---

from PIL import Image

import matplotlib.pyplot as plt

img = Image.open('E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg')

plt.imshow(img)

plt.show()

img_c = img.crop([img.size[0]/4,img.size[1]/4,img.size[0]*3/4,img.size[1]*3/4])

plt.imshow(img_c)

plt.show()
print(type(img.size[0]))
