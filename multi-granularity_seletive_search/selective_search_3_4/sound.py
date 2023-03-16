# -*- coding: utf-8 -*-
# @Time    : 2020/03/07 下午 11:50
# @Author  : Alan D. Chen
# @FileName: sound.py
# @Software: PyCharm

#import winsound
from tkinter import messagebox
import tkinter
import tkinter.messagebox #弹窗库
import tkinter.filedialog
import time

# duration = 5000  # millisecond
# # freq = 700  # Hz
###  蜂鸣声  ###
#winsound.Beep(freq, duration)
stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(stamp)

def theend(l):
    for i in range(l):
        winsound.Beep(220, 600)
        winsound.Beep(330, 600)
        winsound.Beep(440, 600)
        winsound.Beep(550, 600)
        winsound.Beep(660, 600)
        winsound.Beep(770, 600)
        winsound.Beep(880, 600)
        winsound.Beep(990, 600)

    messagebox.showinfo("提示", "您的程序已经运行完毕。" + "\n"
                        + "请您及时记录实验环境、实验参数和实验结果。" + "\n" + stamp)

    # tkinter.messagebox.showinfo('提示', '人生苦短')
    # tkinter.messagebox.showwarning('警告', '明日有大雨')
    # tkinter.messagebox.showerror('错误', '出错了')
    # tkinter.messagebox.askokcancel('提示', '要执行此操作吗')  # 确定/取消，返回值true/false
    # tkinter.messagebox.askquestion('提示', '要执行此操作吗')  # 是/否，返回值yes/no
    # tkinter.messagebox.askyesno('提示', '要执行此操作吗')  # 是/否，返回值true/false
    # tkinter.messagebox.askretrycancel('提示', '要执行此操作吗')  # 重试/取消，返回值true/false

"""
    a = tkinter.filedialog.asksaveasfilename()  # 返回文件名
    print(a)
    a = tkinter.filedialog.asksaveasfile()  # 会创建文件
    print(a)
    a = tkinter.filedialog.askopenfilename()  # 返回文件名
    print(a)
    a = tkinter.filedialog.askopenfile()  # 返回文件流对象
    print(a)
    a = tkinter.filedialog.askdirectory()  # 返回目录名
    print(a)
    a = tkinter.filedialog.askopenfilenames()  # 可以返回多个文件名
    print(a)
    a = tkinter.filedialog.askopenfiles()  # 多个文件流对象
    print(a)
"""
