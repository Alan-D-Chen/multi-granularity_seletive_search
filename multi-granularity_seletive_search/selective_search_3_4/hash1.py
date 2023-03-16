# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: hash1.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,五月 07
# ---

import cv2


# 均值哈希算法

def ahash(image):
    # 将图片缩放为8*8的
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # s为像素和初始灰度值，hash_str为哈希值初始值
    s = 0
    ahash_str = ''
    # 遍历像素累加和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 计算像素平均值
    avg = s / 64
    # 灰度大于平均值为1相反为0，得到图片的平均哈希值，此时得到的hash值为64位的01字符串
    ahash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                ahash_str = ahash_str + '1'
            else:
                ahash_str = ahash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(ahash_str[i: i + 4], 2))
    # print("ahash值：",result)
    return result

# 差异值哈希算法
def dhash(image):
    # 将图片转化为8*8
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''

    for i in range(0, 64, 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
    # print("dhash值",result)
    return result

# 计算两个哈希值之间的差异
def campHash(hash1, hash2):
    n = 0
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1
    # 如果hash长度相同遍历长度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

img1 = "../doraemon/image-003.png"
img2 = "../doraemon/image-019.jpg"
img1 = cv2.imread(img1)
img2 = cv2.imread(img2)
hash1 = ahash(img1)
print('img1的ahash值', hash1)
hash2 = dhash(img1)
print('img1的dhash值', hash2)
hash3 = ahash(img2)
print('img2的ahash值', hash3)
hash4 = dhash(img2)
print('img2的dhash值', hash4)
camphash1 = campHash(hash1, hash3)
camphash2 = campHash(hash2, hash4)
print("ahash均值哈希相似度：", camphash1)
print("dhash差异哈希相似度：", camphash2)
