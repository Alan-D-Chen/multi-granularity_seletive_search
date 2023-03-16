# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: drawing.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,六月 04
# ---

import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

waters = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tv/monitor',)
ABO_score = [6, 7, 6, 1, 2]

plt.barh(waters, ABO_score)  # 横放条形图函数 barh
plt.title('男性购买饮用水情况的调查结果')

plt.show()
