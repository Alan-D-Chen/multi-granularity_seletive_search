# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: 曝光度调整.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,四月 11
# ---
import skimage
import skimage.morphology
import skimage.data
import skimage.io
import skimage.exposure

"""
# 返回直方图, 直方图分成几份，返回两个数组分别是该组别的像素个数和组别
skimage.exposure.histogram(image, nbins=256)

# 直方图均衡化,mask和原图大小一致bool类型只有0或1
skimage.exposure.equalize_hist(image, nbins=256, mask=None)

# 自适应直方图均衡化
skimage.exposure.equalize_adapthist (image, kernel_size=None, clip_limit=0.01, nbins=256)

# 重置强度区间
# image会自动计算图像的最大值和最小值
# 给出数据结构如np.uint8则会把数据归到uint8的区间之内
# 或者给出最大最小值[min, max]
skimage.exposure.rescale_intensity(image, in_range='image', out_range='dtype')

# 返回图像的累积分布和每个分块的中点
skimage.exposure.cumulative_distribution(image, nbins=256)

# 对图像进行gamma调整
# 输入输出都是0到1之间，新像素值Out = In^gamma
skimage.exposure.adjust_gamma(image, gamma=1, gain=1)

# 对图像进行sigmoid纠正，这个操作可能是用在特定领域感觉不太常见
# 公式是Out = 1/(1 + exp*(gain*(cutoff - In)))
# inv如果是True则返回负的sigmoid纠正结果
# 输入输出是0到1之间
skimage.exposure.adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False)

# 对图像进行log调整
# inv为false时Out = gain*log(1 + In)
# inv为true时Out = gain*(2^In - 1)
skimage.exposure.adjust_log(image, gain=1, inv=False)

# 判断图片是否是低分辨率
# 具体的计算手册也并没有写的十分清楚，因此贴图在此，不细描述。
skimage.exposure.is_low_contrast(image, fraction_threshold=0.05, lower_percentile=1, upper_percentile=99, method='linear')
"""

img = skimage.data.immunohistochemistry()
skimage.io.imshow(img)
skimage.io.show()

img_histeq = skimage.exposure.equalize_adapthist (img,20)
skimage.io.imshow(img_histeq)
skimage.io.show()

img_gamma = skimage.exposure.adjust_gamma(img, gamma=0.5, gain=1)
skimage.io.imshow(img_gamma)
skimage.io.show()

img_sigmoid = skimage.exposure.adjust_sigmoid(img)
skimage.io.imshow(img_sigmoid)
skimage.io.show()
