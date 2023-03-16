# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: hash2.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,五月 07
# ---

import numpy as py

import cv2

# 计算hash值
def p_pash(path):
    # 读取图片
    src = cv2.imread(path, 0)
    # 将图片压缩为8*8的，并转化为灰度图
    img = cv2.resize(src, (8, 8), cv2.COLOR_RGB2GRAY)
    # 计算图片的平均灰度值
    avg = sum([sum(img[i]) for i in range(8)]) / 64
    # 计算哈希值,与平均值比较生成01字符串
    str = ''
    for i in range(8):
        str += ''.join(map(lambda i: '0' if i < avg else '1', img[i]))
    # 计算hash值, 将64位的hash值，每4位合成以为，转化为16 位的hash值
    result = ''

    for i in range(0, 64, 4):
        result += ''.join('%x' % int(str[i: i + 4], 2))
    print(result)
    return result

# 计算汉明距离
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        return
    count = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            count += 1
    return count

h1 = p_pash('../cat1/cat.1.jpg')
h2 = p_pash('../doraemon/image-003.png')
h3 = p_pash('../doraemon/image-019.jpg')
print(hamming_distance(h1, h2))
print(hamming_distance(h1, h3))
print(hamming_distance(h2, h3))