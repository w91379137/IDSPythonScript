#-*- coding: utf-8 -*-
#-*- coding: cp950 -*-

import os,sys
from pylab import *
from PIL import Image

#-----.-----.-----.-----.-----.
import PIL as pl
import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#-----.-----.-----.-----.-----.
import ServerKit as se

#-----.-----.-----.-----.-----.
img_ref = pl.Image.open('/Users/w91379137/Desktop/python/ServerTest/ref.png')

img_target = pl.Image.open('/Users/w91379137/Desktop/python/ServerTest/target.png')

"""
    Test edge_tools
"""
#將圖片亮度 調整
#img, _ = se.edge_tools.histeq(img)

#標記出邊緣線
#https://github.com/jterrace/pyssim/issues/6
#img, _, _ = se.edge_tools.find_edge_gradient(pl.Image.fromarray(img))

#標記出邊緣線
#img = se.edge_tools.find_edge_canny(pl.Image.fromarray(img))

"""
    Test segment_tools
"""

#取得臉部座標
#a,b = se.segment_tools.faces_positions(pl.Image.fromarray(img))

"""
    Test face_align_tools
"""
#標記出 重要點位置
#img = se.face_align_tools.annotate_face(pl.Image.fromarray(img_modify))

#將圖片位置 轉換到 目標圖片位置
#img = se.face_align_tools.face_align(img_ref,img_target)

"""
    書本測試
"""
#box = (100,100,200,200)
#region_img = img_ref.crop(box)
#region_img = region_img.transpose(pl.Image.ROTATE_180)
#img_ref.paste(region_img,box)

#img_ref = imgArr_ref.resize((50,50))
#img_ref = imgArr_ref.rotate(45)


#P60
im1 = sp.ndimage.imread('/Users/w91379137/Desktop/python/ServerTest/ref.png')
m, n = im1.shape[:2]

tp = np.array([[675,826,826,677],[55,52,281,277],[1,1,1,1]])
fp = np.array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

tp2 = tp[:,:3]
fp2 = fp[:,:3]





"""
    輸出
"""

#plt.imshow(img)
#plt.xticks([]), plt.yticks([])
#plt.show()
