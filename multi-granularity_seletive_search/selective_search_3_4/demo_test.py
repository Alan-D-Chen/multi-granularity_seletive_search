# -*- coding: utf-8 -*-
# @Time    : 2020/03/19 下午 10:20
# @Author  : Alan D. Chen
# @FileName: demo_test.py
# @Software: PyCharm


# -*- coding: utf-8 -*-

from __future__ import (
    division,
    print_function,
)
from skimage import io, data
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
import sys
#import pysnooper
import time
import sound
import os
import cv2
from selective_search import selective_search
from xml_extract import xml_extract
from IOU2 import calIOU_V2
# import PIL.Image

sys.path.append('C:/Users/陈冬/Pictures/Camera Roll/')
path = "E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
# path = paths
path2 = "E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations"
result = os.listdir(path)
###  遍历多少张图片  ###
num = 5
stampss = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def main():
    list3 = []
    results = []
    results2 = []
    results3 = []
    results4 = []
    results5 = []
    results6 = []
    """

    """
    # loading astronaut image
    # img = skimage.data.astronaut()
    # print(img)
    # print("size of pics:", ima.size)
    # io.imshow(ima)
    # img = np.asarray(ima)
    # print("narray of image: \n", img)
    # print("len of img:",len(img))

    # perform selective search
    # ###########
    # print("展示 S 的结构:\n")
    # for k in S.keys():
    #     print('key = {}'.format(k))
    # for v in S.values():
    #     print('values = {}'.format(v))
    # for v, k in S.items():
    #     print('{v}:{k}'.format(v=v, k=k))
    # 计算一共分割了多少个原始候选区域

    ####创建recording#####
    desktop_path = "E:/pycharm-items-github/selective_search_3_4/recording/"  # 新创建的txt文件的存放路径
    name = "Sturding_Recording" + stampss
    full_path = desktop_path + name + ".txt"  # 也可以创建一个文档
    file = open(full_path, 'w')
    file.write("## VOC2012_images_Sturding_Recording ##" + "\n\n")
    # file.close()

    js = 0
    for file in result:
        js = js + 1
        if js == num:
            break
        #print("file:",file,"type(file):",type(file))
        file2 = file[:11]
        #print(file2)
        img = Image.open(path +"/" + file)
        img2 = cv2.imread(path +"/" + file)
        #print("type of img:",type(img))
        img_lbl, regions = selective_search(img,img2, scale=500, sigma=0.9, min_size=10)
        # print("length of region:",len(regions))
        # print("type of regions:",type(regions),"type of regions[0]:",type(regions[1]))
        # print("regions:",regions[1])
        candidates = set()  # 创建一个集合 元素不会重复，每一个元素都是一个list(左上角x，左上角y,宽,高)，表示一个候选区域的边框

        for r in regions:
            # excluding same rectangle (with different segments)         #排除重复的候选区
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels         #排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)
            if r['size'] < 1000:
                continue
            # distorted rects          #排除扭曲的候选区域边框  即只保留近似正方形的
            x, y, w, h = r['rect']
            if w / h > 3.0 or h / w > 3.0:
                continue
            candidates.add(r['rect'])
        """
        # draw rectangles on the original image     #在原始图像上绘制候选区域边框
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(img)
        #print("##### the ", js ,"th  picture #####","Candidate Region Proposal:")
        i = 0
        for x, y, w, h in candidates:
            i = i + 1
            #print("The",i,"th Candidate Region Proposal:", x, y, w, h)
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='blue', linewidth=1)
            ax.add_patch(rect)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        stamps = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # plt.title("SELECTIVE SEARCH RESULTS" + "\n" + file + "\n" + stamp)
        # plt.savefig(
        #     "E:/pycharm-items-github/selective_search_3/pics/selective_search" + file + stamps + ".png")
        # plt.show()
        # plt.cla()
        # plt.close("all")
        """
        #print("longth of candidate regions:",len(candidates))
        #print("candidate regions:\n",candidates,"\n type of candidate regions",type(candidates))

        ######## 提取 groud truth  ###########
        list3 = xml_extract(path +"/" + file , path2 +"/" + file2 + ".xml")
        # print("##### the ", js, "th  picture", "ground truth:")
        # print("list3:\n",list3)

        ###  计算 candidate region proposal 和 ground truth 的IOU ###
        hs = 0
        ls = 0
        results4 = []
        results5 = []
        lt_candidate = list(candidates)
        #print("lt_candidate:\n",lt_candidate)
        #print("type of lt_candidate:",type(lt_candidate))
        for rects in list3:
            results = []
            #results2 = []
            #results3 = []
            rects2 = tuple(rects)
            hs = hs + 1
            #print("rects2:",rects2,"type of rects2:",type(rects2))
            for candis in lt_candidate:
                ls = ls + 1
                candis = list(candis)
                candis = tuple([candis[0],candis[1],candis[2]+candis[0],candis[3]+candis[1]])
                #print("candis:", candis, "type of candis:", type(candis))
                rlt = calIOU_V2(rects2,candis)
                #print("The ground truth:",hs,"--candidate region proposal:",ls,"--IOU:",rlt)
                results.append(rlt)
                results4.append(rlt)
                results5.append(rlt)
            #print("The ground truth:", hs,"result of IOU:\n",results)
            results2.append(max(results))
            results6.append(max(results))
            results3.append(np.mean(results))
            # print("For the ground truth:", hs,"Max result of IOU:\n",max(results))
            # print("For the ground truth:", hs,"Avg result of IOU:\n",np.mean(results))
        # print("For this picture ", js, ", Max result of IOU:",max(results4))
        # print("For this picture ", js, ", Avg result of IOU:",np.mean(results4))
        # print("For this picture ", js, ", Max (ground truth) result of IOU:", max(results2))
        # print("For this picture ", js, ", Avg (ground truth) result of IOU:", np.mean(results2))
    # print("For these pictures , Max result of IOU:", max(results5))
    # print("For these pictures , Avg result of IOU:", np.mean(results5))
    # print("For these pictures , Max (ground truth) result of IOU:", max(results6))
    # print("For these pictures , Avg (ground truth) result of IOU:", np.mean(results6))
    with open(full_path, "a") as f:
        f.write(path + "\n\n" + "For this picture " + str(js) + ", Max result of IOU:" + str(max(results4)) + "\n")
        f.write("For this picture " + str(js) + ", Avg result of IOU:" + str(np.mean(results4)) + "\n")
        f.write("For this picture " + str(js) + ", Max (ground truth) result of IOU:" + str(max(results2)) + "\n")
        f.write("For this picture " + str(js) + ", Avg (ground truth) result of IOU:" + str(np.mean(results2)) + "\n")
        f.write("For these pictures , Max result of IOU:" + str(max(results5)) + "\n")
        f.write("For these pictures , Avg result of IOU:" + str(np.mean(results5)) + "\n")
        f.write("For these pictures , Max (ground truth) result of IOU:" + str(max(results6)) + "\n")
        f.write("For these pictures , Avg (ground truth) result of IOU:" + str(np.mean(results6)))

    """代码运行结束提示"""
    sound.theend(1)

if __name__ == "__main__":
    main()
