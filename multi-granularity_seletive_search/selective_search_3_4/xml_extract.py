#!/usr/bin/python
# -*- coding: UTF-8 -*-
# get annotation object bndbox location
import os
import cv2

try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET


##get object annotation bndbox loc start
def GetAnnotBoxLoc(AnotPath):  # AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet = {}  # 以目标类别为关键字，目标框为值组成的字典结构
    list1= []
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)  # -1
        x2 = int(BndBox.find('xmax').text)  # -1
        y2 = int(BndBox.find('ymax').text)  # -1
        BndBoxLoc = [x1, y1, x2, y2]
        list1.append(BndBoxLoc)
        if ObjName in ObjBndBoxSet:
            ObjBndBoxSet[ObjName].append(BndBoxLoc)  # 如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
            ObjBndBoxSet[ObjName] = [BndBoxLoc]  # 如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
    return (ObjBndBoxSet,list1)


##get object annotation bndbox loc end

def display(objBox, pic):
    img = cv2.imread(pic)

    for key in objBox.keys():
        for i in range(len(objBox[key])):
            cv2.rectangle(img, (objBox[key][i][0], objBox[key][i][1]), (objBox[key][i][2], objBox[key][i][3]),
                          (0, 0, 255), 2)
            cv2.putText(img, key, (objBox[key][i][0], objBox[key][i][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
    cv2.imshow('img', img)
    cv2.imwrite('display.jpg', img)
    cv2.waitKey(0)


#if __name__ == '__main__':
def xml_extract(path1 , path2):
    list2 = []
    #pic = r"E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"
    #ObjBndBoxSet,list2  = GetAnnotBoxLoc(r"E:/dataset_1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/2007_000032.xml")
    pic = path1
    ObjBndBoxSet, list2 = GetAnnotBoxLoc(path2)

    # print(type(ObjBndBoxSet))
    # print(ObjBndBoxSet)
    # print(type(list2))
    # print(list2)
    #display(ObjBndBoxSet, pic)
    return list2
