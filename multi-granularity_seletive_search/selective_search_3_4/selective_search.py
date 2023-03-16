# -*- coding: utf-8 -*-
# @Time    : 2020/03/19 下午 10:21
# @Author  : Alan D. Chen
# @FileName: selective_search.py
# @Software: PyCharm

from __future__ import division
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy
import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology,color,data,filters

# "Selective Search for Object Recognition" by J.R.R. Uijlings et al.
#  - Modified version with LBP extractor for texture vectorization

def _generate_segments(im_orig, scale, sigma, min_size):
    """
        segment smallest regions by the algorithm of Felzenswalb and
        Huttenlocher
    """
    # 对图像进行分块处理，作为rgb的第四维，为region
    # open the Image
    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)
    # print("im_orig.shape:\n",im_orig.shape)
    # print("im_orig.shape[:2]: \n", im_orig.shape[:2])
    # print("im_mask.shape: \n", im_mask.shape)
    # merge mask channel to the image as a 4th channel
    im_orig = numpy.append(im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask
    #print("im_orig.shape: \n",im_orig.shape)

    return im_orig

def p_hash(img):
    # 将图片压缩为8*8的，并转化为灰度图
    img = cv2.resize(np.float32(img), (8, 8), cv2.COLOR_RGB2GRAY)
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
    # print(result)
    return result

# 计算汉明距离
def hamming_distance(r1, r2):
    #print(r1)
    str1 = r1["hash_o"]
    str2 = r2["hash_o"]
    # if len(str1) != len(str2):
    #     return count1
    count1 = 0
    for i in range(len(str2)):
        if str1[i] == str2[i]:
            count1 += 1

    str1 = r1["hash_o2"]
    str2 = r2["hash_o2"]
    # if len(str1) != len(str2):
    #     return count2
    count2 = 0
    for i in range(len(str2)):
        if str1[i] == str2[i]:
            count2 += 1

    str1 = r1["hash_o3"]
    str2 = r2["hash_o3"]
    # if len(str1) != len(str2):
    #     return count3
    count3 = 0
    for i in range(len(str2)):
        if str1[i] == str2[i]:
            count3 += 1
    # print("count1, count2, count3",count1, count2, count3)
    return (count1, count2, count3)

def _sim_colour(r1, r2):
    """
    计算颜色相似度
    calculate the sum of histogram intersection of colour
    args:
        r1：候选区域r1
        r2：候选区域r2
    return：[0,3]之间的数值
    """
    return (sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])]),
           sum([min(a, b) for a, b in zip(r1["hist_c2"], r2["hist_c2"])]),
           sum([min(a, b) for a, b in zip(r1["hist_c3"], r2["hist_c3"])]),
           sum([min(a, b) for a, b in zip(r1["hist_c4"], r2["hist_c4"])]),
           sum([min(a, b) for a, b in zip(r1["hist_c5"], r2["hist_c5"])]))


def _sim_texture(r1, r2):
    """
    计算纹理特征相似度
    calculate the sum of histogram intersection of texture
    args:
        r1：候选区域r1
        r2：候选区域r2
    return：[0,3]之间的数值
    """
    return (sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])]),
           sum([min(a, b) for a, b in zip(r1["hist_t2"], r2["hist_t2"])]),
           sum([min(a, b) for a, b in zip(r1["hist_t3"], r2["hist_t3"])]),
           sum([min(a, b) for a, b in zip(r1["hist_t4"], r2["hist_t4"])]),
           sum([min(a, b) for a, b in zip(r1["hist_t5"], r2["hist_t5"])]))


def _sim_size(r1, r2, imsize):
    """
    计算候选区域大小相似度
    calculate the size similarity over the image
    args:
        r1：候选区域r1
        r2：候选区域r2
    return：[0,1]之间的数值
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
    计算候选区域的距离合适度相似度
    calculate the fill similarity over the image
    args:
        r1：候选区域r1
        r2：候选区域r2
        imsize：原图像像素数
    return：[0,1]之间的数值
    """
    bbsize = (
            (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
            * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    '''
        计算两个候选区域的相似度，权重系数默认都是1
        args:
            r1：候选区域r1
            r2：候选区域r2
            imsize：原图片像素数
    '''
    #系数 a 1 2 3 4##
    sim_c, sim_c2, sim_c3, sim_c4, sim_c5 = _sim_colour(r1, r2)
    sim_t, sim_t2, sim_t3, sim_t4, sim_t5 = _sim_texture(r1, r2)
    sim_o, sim_o2, sim_o3 = hamming_distance(r1, r2)

    # print("sim_c, sim_c2, sim_c3, sim_c4, sim_c5, "
    #       "sim_t, sim_t2, sim_t3, sim_t4, sim_t5,sim_o, sim_o2, sim_o3:\n",sim_c, sim_c2,
    #       sim_c3, sim_c4, sim_c5,sim_t, sim_t2, sim_t3, sim_t4, sim_t5,sim_o, sim_o2, sim_o3)
    return (5*(sim_c + 2*sim_c2 + sim_c3 + sim_c4 + sim_c5) + 2*(sim_t + 5*sim_t2 + 1*sim_t3 + 5*sim_t4 + 2*sim_t5)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize) + (sim_o + sim_o2 + sim_o3))


def _calc_colour_hist(img):
    """
    使用L1-norm归一化获取图像每个颜色通道的25 bins的直方图，这样每个区域都可以得到一个75维的向量
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    args:
        img：ndarray类型， 形状为候选区域像素数 x 3(h,s,v)
    return：一维的ndarray类型，长度为75
    """
    ##############  在这里进行修改color的计算  ##############
    BINS = 25
    hist = numpy.array([])
    for colour_channel in (0, 1, 2):
        # extracting one colour channel
        c = img[:, colour_channel]
        # calculate histogram for each colour and join to the result
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1 normalize len(img):候选区域像素数
    #print("hist: \n", hist)
    hist = hist / len(img)
    #print("hist/len(img): \n",hist)
    ###########增加不同色彩空间的图片的色彩直方图##########

    return hist



def _calc_texture_gradient(img):
    """
    原文：对每个颜色通道的8个不同方向计算方差σ=1的高斯微分（Gaussian Derivative，这里使用LBP替代
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we use LBP instead.
    output will be [height(*)][width(*)]
    args：
        img： ndarray类型，形状为height x width x 4，每一个像素的值为 [r,g,b,(region)]
    return：纹理特征，形状为height x width x 4
    """
    ret = numpy.zeros((img.shape[0], img.shape[1], img.shape[2]))
    #print("img.shape[0], img.shape[1], img.shape[2]: \n",img.shape[0], img.shape[1], img.shape[2])
    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(img[:, :, colour_channel], 8, 1.0)
    #print("ret: \n",ret)
    return ret

def _calc_texture_gradient2(img):
    """
    原文：对每个颜色通道的8个不同方向计算方差σ=1的高斯微分（Gaussian Derivative，这里使用LBP替代
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we use LBP instead.
    output will be [height(*)][width(*)]
    args：
        img： ndarray类型，形状为height x width x 4，每一个像素的值为 [r,g,b,(region)]
    return：纹理特征，形状为height x width x 4
    """
    ret = numpy.zeros((img.shape[0], img.shape[1]))
    #print("img.shape[0], img.shape[1]: \n",img.shape[0], img.shape[1])
    #for colour_channel in (0):
    ret[:, :] = skimage.feature.local_binary_pattern(img[:, :], 8, 1.0)
    #print("ret: \n",ret)
    return ret

def _calc_texture_hist(img):
    """
        calculate texture histogram for each region
        calculate the histogram of gradient for each colours
        the size of output histogram will be
            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """

    BINS = 10
    hist = numpy.array([])
    #print("img.shape:",img.shape)

    for colour_channel in (0, 1, 2):
        # mask by the colour channel
        fd = img[:, colour_channel]
        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])
    # L1 Normalize
    #print("hist-2: \n", hist)
    hist = hist / len(img)
    #print("hist-2/len(img): \n", hist)
    return hist
def _calc_texture_hist2(img):
    """
        calculate texture histogram for each region
        calculate the histogram of gradient for each colours
        the size of output histogram will be
            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """

    BINS = 10
    hist = numpy.array([])
    #print("img.shape:",img.shape)

    #for colour_channel in (0, 1, 2):
        # mask by the colour channel
    fd = img[:]
        # calculate histogram for each orientation and concatenate them all
        # and join to the result
    hist = numpy.concatenate(
            [hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])
    # L1 Normalize
    #print("hist-2: \n", hist)
    hist = hist / len(img)
    #print("hist-2/len(img): \n", hist)
    return hist


def _extract_regions(img):
    '''
        提取每一个候选区域的信息 ##  比如类别(region)为5的区域表示的是一只猫的选区，这里就是提取这只猫的边界框，左上角后右下角坐标
        args:
            img: ndarray类型，形状为height x width x 4，每一个像素的值为 [r,g,b,(region)]
        return :
            R:dict 每一个元素对应一个候选区域， 每个元素也是一个dict类型
                                  {min_x:边界框的左上角x坐标,
                                  min_y:边界框的左上角y坐标,
                                  max_x:边界框的右下角x坐标,
                                  max_y:边界框的右下角y坐标,
                                  size:像素个数,
                                  hist_c:颜色的直方图,hist_c(2~5):
                                  hist_t:纹理特征的直方图,
                                  hash_o:轮廓特征的计算方法}
    '''

    R = {}
    #print("type of R:",type(R),"R:",R)
    # get hsv\lab\luv\yuv\rgb images for colors
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
    lab = skimage.color.rgb2lab(img[:, :, :3])
    luv = skimage.color.rgb2luv(img[:, :, :3])
    yuv = skimage.color.rgb2yuv(img[:, :, :3])
    rgb = img[:, :, :3]
    # get images for texture
    image2 = cv2.Laplacian(rgb, cv2.CV_64F)
    binf = cv2.inRange(rgb, np.array([100, 43, 46]), np.array([124, 255, 255]))
    binf2 = cv2.inRange(rgb, np.array([90, 43, 46]), np.array([130, 255, 255]))
    imgs3 = Image.fromarray(rgb.astype('uint8')).convert('RGB')
    imgs = ImageEnhance.Contrast(imgs3).enhance(2.0)
    imgs2 = ImageEnhance.Sharpness(imgs).enhance(2.0)
    gray = cv2.cvtColor(np.float32(imgs3), cv2.COLOR_RGB2GRAY)
    gray = np.asarray(gray)
    """
    # get images for outlines
    #grayx = skimage.color.rgb2gray(img)
    # grayx = gray/255
    # denoised = filters.rank.median(grayx, morphology.disk(2))  # 过滤噪声
    #         # 将梯度值低于10的作为开始标记点
    # markers = filters.rank.gradient(denoised, morphology.disk(5)) < 10
    # markers = ndi.label(markers)[0]
    # gradient = filters.rank.gradient(denoised, morphology.disk(2))  # 计算梯度
    # segmen = morphology.watershed(gradient, markers, mask=grayx)  # 基于梯度的分水岭算法
    # print("markers.shape,gradient.shape,labels.shape",markers.shape,gradient.shape,labels.shape)
    # cv2.imshow("img",img)
    # cv2.imshow("hsv",hsv)
    # cv2.imshow("lab",lab)
    # cv2.imshow("luv",luv)
    # cv2.imshow("yuv",yuv)
    # cv2.imshow("rgb",rgb)
    # cv2.imshow("gray",gray)
    # cv2.imshow("binf",binf)
    # cv2.imshow("binf2",binf2)
    # cv2.imshow("image2",image2)
    # cv2.imshow("imgs",imgs3)
    # cv2.imshow("markers",markers)
    # cv2.imshow("gradient",gradient)
    # cv2.imshow("labels",labels)
    # cv2.waitKey(0)
    # imgs.show()
    # imgs2.show()
    """

# pass 1: count pixel positions 遍历每一个像素
    for y, i in enumerate(img):
        #print("y: \n",y)
        #print("i.shape: \n",i.shape)
        for x, (r, g, b, l) in enumerate(i):
            # initialize a new region  #为什么要初始化最大##
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}
            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

            #print("Detail of Region Proposal: \n",
                 #R[l]["min_x"], R[l]["min_y"], R[l]["max_x"], R[l]["max_y"])

    # pass 2: calculate texture gradient
    # 纹理特征提取 利用LBP算子 height x width x 4
        """
        print("img.ndim -image2.ndim -binf.ndim -binf2.ndim -gray.ndim :\n",
            img.ndim, image2.ndim, binf.ndim, binf2.ndim, gray.ndim,
            "\n img.ndim -image2.ndim -binf.ndim -binf2.ndim -gray.ndim :\n",
            img.shape, image2.shape, binf.shape, binf2.shape, gray.shape)
        """
    tex_grad = _calc_texture_gradient(img)
    tex_grad2 = _calc_texture_gradient(image2)
    tex_grad3 = _calc_texture_gradient2(binf)
    tex_grad4 = _calc_texture_gradient2(binf2)
    tex_grad5 = _calc_texture_gradient2(gray)

    # pass 3: calculate colour histogram of each region
    # 计算每一个候选区域(注意不是bounding box圈住的区域)的直方图
    #print("R.items(): \n",R.items())
    for k, v in list(R.items()):
        # print(hsv.shape)(500, 666, 3)
        # print(hsv[:, :, :].shape)
        # print(img.shape) (500, 666, 4)
        # print(img[:, :, 3].shape)
        #print("img[:, :, 3]:\n",img[:, :, 3])

        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        masked_pixels2 = lab[:, :, :][img[:, :, 3] == k]
        masked_pixels3 = luv[:, :, :][img[:, :, 3] == k]
        masked_pixels4 = yuv[:, :, :][img[:, :, 3] == k]
        masked_pixels5 = rgb[:, :, :][img[:, :, 3] == k]
        #print("masked_pixels.shape: \n",masked_pixels.shape)
        R[k]["size"] = len( masked_pixels/ 4)  # 为什么除以4 计算长度 #候选区域k像素数???
        # 在hsv色彩空间下，使用L1-norm归一化获取图像每个颜色通道的25 bins的直方图，这样每个区域都可以得到一个75维的向量
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)
        R[k]["hist_c2"] = _calc_colour_hist(masked_pixels2)
        R[k]["hist_c3"] = _calc_colour_hist(masked_pixels3)
        R[k]["hist_c4"] = _calc_colour_hist(masked_pixels4)
        R[k]["hist_c5"] = _calc_colour_hist(masked_pixels5)
        #print("tex_grad[:, :].shape:\n",tex_grad[:, :].shape)
        ############ 图像对比度和清晰度的提升 #################
        # texture histogram
        # 在rgb色彩空间下，使用L1-norm归一化获取图像每个颜色通道的每个方向的10 bins的直方图，这样就可以获取到一个240（10x8x3）维的向量
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])
        R[k]["hist_t2"] = _calc_texture_hist(tex_grad2[:, :][img[:, :, 3] == k])
        R[k]["hist_t3"] = _calc_texture_hist2(tex_grad3[:, :][img[:, :, 3] == k])
        R[k]["hist_t4"] = _calc_texture_hist2(tex_grad4[:, :][img[:, :, 3] == k])
        R[k]["hist_t5"] = _calc_texture_hist2(tex_grad5[:, :][img[:, :, 3] == k])
        # outlines algorithm
        # R[k]["hash_o"] = p_hash(gradient[:, :][img[:, :, 3] == k])
        # R[k]["hash_o2"] = p_hash(markers[:, :][img[:, :, 3] == k])
        # R[k]["hash_o3"] = p_hash(segmen[:, :][img[:, :, 3] == k])
        R[k]["hash_o"] = "00"
        R[k]["hash_o2"] = "00"
        R[k]["hash_o3"] = "00"

    ########展示 R的结构  ##########
    # js = 0
    # print("R:\n", "################### key: ; values： ","####################")
    # for v, k in R.items():
    #     #if js == 1:
    #         print("R:", js, "~key: ; values: ------> \n", '{v}:{k}'.format(v=v, k=k))
    #         js = js + 1
    #         break

    return R


def _extract_neighbours(regions):
    '''
        提取 邻居候选区域对(ri,rj)(即两两相交)
        args:
            regions：dict 每一个元素都对应一个候选区域
        return：
            返回一个list，每一个元素都对应一个邻居候选区域对
    '''

    # 判断两个候选区域是否相交

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
            and a["min_y"] < b["min_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # 转换为list 每一个元素 (l,regions[l])
    #print("type of regions.items:\n",type(regions.items))
    R = list(regions.items())
    #print("R:\n", R)
    # 保存两两相交候选区域对
    neighbours = []

    # 每次抽取两个候选区域 两两组合，判断是否相交
    for cur, a in enumerate(R[:-1]):
        # print("cur:",cur)
        # print("a:\n",a)
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours

### 合并regions #######
def _merge_regions(r1, r2, markers, gradient, labels):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"])
    }
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_c2": (r1["hist_c2"] * r1["size"] + r2["hist_c2"] * r2["size"]) / new_size,
        "hist_c3": (r1["hist_c3"] * r1["size"] + r2["hist_c3"] * r2["size"]) / new_size,
        "hist_c4": (r1["hist_c4"] * r1["size"] + r2["hist_c4"] * r2["size"]) / new_size,
        "hist_c5": (r1["hist_c5"] * r1["size"] + r2["hist_c5"] * r2["size"]) / new_size,
        "hist_t": (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "hist_t2": (r1["hist_t2"] * r1["size"] + r2["hist_t2"] * r2["size"]) / new_size,
        "hist_t3": (r1["hist_t3"] * r1["size"] + r2["hist_t3"] * r2["size"]) / new_size,
        "hist_t4": (r1["hist_t4"] * r1["size"] + r2["hist_t4"] * r2["size"]) / new_size,
        "hist_t5": (r1["hist_t5"] * r1["size"] + r2["hist_t5"] * r2["size"]) / new_size,
        "hash_o": p_hash(markers.crop([rt["min_x"], rt["min_y"], rt["max_x"], rt["max_y"]])),
        "hash_o2": p_hash(gradient.crop([rt["min_x"], rt["min_y"], rt["max_x"], rt["max_y"]])),
        "hash_o3": p_hash(labels.crop([rt["min_x"], rt["min_y"], rt["max_x"], rt["max_y"]])),
        "labels": r1["labels"] + r2["labels"],
    }

    return rt
#@pysnooper.snoop()

def selective_search(im_orig,im_orig2, scale=1.0, sigma=0.8, min_size=50):
    '''Selective Search
    Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int    用felzenszwalb segmentation算法把图像簇的个数
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float 表示felzenszwalb分割时，用的高斯核宽度 相当于是方差
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int 表示分割后最小组尺寸
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''
    ima = np.asarray(im_orig)
    print("shape of ima:", ima.shape)
    print("ima:", ima)
    assert ima.shape[2] == 3, "3ch image is expected"
    # load image and get smallest regions
    # region label is stored in the 4th value of each pixel [r,g,b,(region)]

    # Branch-2  just for watershed
    #ima2 = np.asarray(im_orig)#, dtype=np.uint8)
    print("im_orig2:-->", im_orig2)
    print("shape of im_orig2:", im_orig2.shape)
    print("type of im_orig2:", type(im_orig2))
    imagex = color.rgb2gray(im_orig2)
    denoised = filters.rank.median(imagex, morphology.disk(2))  # 过滤噪声
    # 将梯度值低于10的作为开始标记点
    markers = filters.rank.gradient(denoised, morphology.disk(5)) < 10
    markers = ndi.label(markers)[0]
    gradient = filters.rank.gradient(denoised, morphology.disk(2))  # 计算梯度
    labels = morphology.watershed(gradient, markers, mask=imagex)  # 基于梯度的分水岭算法
    markers = Image.fromarray(markers)
    gradient = Image.fromarray(gradient)
    labels = Image.fromarray(labels)
    # cv2.imshow("gradient", gradient)
    # cv2.imshow("markers", markers)
    # cv2.imshow("labels", labels)
    # cv2.waitKey(0)

    img = _generate_segments(ima, scale, sigma, min_size)
############################################################
    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img)
    # print(R[199]["hash_o"])

    # extract neighbouring information 每一个元素都是邻居候选区域对(ri,rj)  (即两两相交的候选区域)
    neighbours = _extract_neighbours(R)

    # calculate initial similarities 初始化相似集合S = ϕ
    S = {}

    # 计算每一个邻居候选区域对的相似度s(ri,rj)
    for (ai, ar), (bi, br) in neighbours:
        # S=S∪s(ri,rj)  ai表示候选区域ar的标签
        # 比如当ai=1 bi=2 S[(1,2)就表示候选区域1和候选区域2的相似度
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    """
    # js = 0
    #     ################# 输出 S 的结构 ####################
    # print("S:\n", "################### key: ; values： ", "####################")
    # for v, k in S.items():
    #     print("S:", js, "~key: ; values: ------> \n ", '{v}:{k}'.format(v=v, k=k))
    #     js = js + 1
    #     break
    """
    # hierarchal search 层次搜索 直至相似度集合为空
    while S != {}:
        # get highest similarity  获取相似度最高的两个候选区域  i,j表示候选区域标签
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]
        #print("sorted(S.items(), key=lambda i: i[1]):\n",sorted(S.items(), key=lambda i: i[1]))  # 按照相似度排序
        # merge corresponding regions 合并相似度最高的两个邻居候选区域 rt = ri∪rj ,R = R∪rt
        t = max(R.keys()) + 1.0
        #######################################
        R[t] = _merge_regions(R[i], R[j], markers, gradient, labels)
        # mark similarities for regions to be removed 获取需要删除的元素的键值
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)
        # remove old similarities of related regions
        # 移除候选区域ri对应的所有相似度：S = S\s(ri,r*)  移除候选区域rj对应的所有相似度：S = S\s(r*,rj)
        for k in key_to_delete:  # k表示邻居候选区域对(i,j)  v表示候选区域(i,j)表示相似度
            del S[k]
        # calculate similarity set with the new region
        # 计算候选区域rt对应的相似度集合St,S = S∪St
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)


    # 获取每一个候选区域的的信息  边框、以及候选区域size,标签
    regions = []
    for k, r in list(R.items()):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })
    # img：基于图的图像分割得到的候选区域
    # regions：Selective Search算法得到的候选区域

    return img, regions
