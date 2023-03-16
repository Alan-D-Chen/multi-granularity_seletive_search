# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: IOU2.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2020,四月 23
# ---

def calIOU_V1(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # 计算每个矩形的面积
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    # judge if there is an intersect

    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)

def calIOU_V2(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # cx1 = rec1[0]
    # cy1 = rec1[1]
    # cx2 = rec1[2]
    # cy2 = rec1[3]
    # gx1 = rec2[0]
    # gy1 = rec2[1]
    # gx2 = rec2[2]
    # gy2 = rec2[3]
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    # 计算每个矩形的面积
    S_rec1 = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    S_rec2 = (gx2 - gx1) * (gy2 - gy1)  # G的面积
    # 计算相交矩形
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积
    iou = area / (S_rec1 + S_rec2 - area)
    return iou

# if __name__ == '__main__':
#     rect1 = (661, 27, 679, 47)
#     # (top, left, bottom, right)
#     rect2 = (662, 27, 682, 47)
#     print("type of rect1:",type(rect1),"length of rect1:",len(rect1))
#     iou1 = calIOU_V1(rect1, rect2)
#     iou2 = calIOU_V2(rect1, rect2)
#     print("iou1:",iou1)
#     print("iou2:",iou2)


