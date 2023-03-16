# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: drawing_2.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,六月 05
# ---
# seaborn模块之垂直或水平条形图
# 导入第三方模块
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 读入数据
GDP = pd.read_excel('C:/Users/陈冬/Desktop/SS_LTC/recording_3.xlsx')
sns.barplot(y = 'Category', # 指定条形图x轴的数据
            x = 'ABO', # 指定条形图y轴的数据
            data = GDP, # 指定需要绘图的数据集
            color = 'steelblue', # 指定条形图的填充色
            orient = 'horizontal' # 将条形图水平显示recording_3.xlsx
           )
# 重新设置x轴和y轴的标签
plt.xlabel('GDP（万亿）')
plt.ylabel('')
# 添加图形的标题
plt.title('2017年度6个省份GDP分布')
# 为每个条形图添加数值标签
for y,x in enumerate(GDP.ABO):
     plt.text(x,y,'%s' %round(x,4),va='center')
# 显示图形
plt.show()