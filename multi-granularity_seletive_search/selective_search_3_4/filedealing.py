# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: filedealing.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,五月 28
# ---

import os
import shutil
#from demo_test import mains

path5 = 'E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main'
js = 0

num = 63
num2 = 2000

result = os.listdir(path5)

#####创建recording#####
desktop_path = "E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/images/"  # 新创建的txt文件的存放路径
name = "VOC2012_images_recording"
full_path = desktop_path + name + ".txt"  # 也可以创建一个.doc的word文档
file = open(full_path, 'w')
file.write("## VOC2012_images_recording ##" + "\n\n")  # msg也就是下面的Hello world!
# file.close()

for file in result:

    if js == num:
        break
    #print(file + " ########## " + file[:-4])
    #print(file[-9:-4],type(file))
    if str(file[-9:-4]) == "train":
        os.mkdir("E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/images/" + file[:-4])
        path2 = "E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/images/" + file[:-4]
        path3 = "E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/"
        path4 = path5 + "/" + file

        file2 = open(path4)
        #print(file + "######" + path4)

        ls = 0
        hs = 0
        for line in file2:

            if ls == num2:
                break
            #print("@@@@@@" + line[-3:-1])
            if line[-3:-1] == " 1":
                #print("####-right-####" + line)
                shutil.copy(path3 + line[:-4] + ".jpg", path2)
                hs = hs + 1
            ls = ls + 1

        with open(full_path, "a") as f:
            f.write(path2 + "\n" + "There are " + str(hs) + " images.\n" + "\n")


    # paths = path2
    # print(paths)
    # mains(paths)
    js = js + 1
    #file.close()

