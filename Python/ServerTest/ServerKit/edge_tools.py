#-*- coding: utf-8 -*-
#-*- coding: cp950 -*-

from PIL import Image
from pylab import *
import pandas as pd
from scipy.ndimage import filters
import sys, os
import cv2
import numpy as np

def histeq(im_data, nbr_bins = 256):
    """
        Histogram equalization of a grayscale image.
        用來消除圖片其中因為亮度
        平均顏色分佈 增加全局對比
    """
    
    #http://stackoverflow.com/questions/14801923/histogram-equalization-for-python
    
    #圖值 imhist
    #分布圖 bins
    imhist, bins = histogram(im_data.flatten(), nbr_bins, density = True)
    cdf = imhist.cumsum()
    
    #累積分布圖 cdf
    cdf = 255 * cdf / cdf[-1] #除上最後一個 累積出來不一定剛好是1 所以標準化
    
    im2 = interp(im_data.flatten(), bins[:-1], cdf)
    return np.floor(im2.reshape(im_data.shape)), cdf

def summary(im):
    """
        圖片的概略資訊 pandas
        count    187500.000000
        mean        106.803632
        std         110.490913
        min           0.000000
        25%           0.000000
        50%          32.000000
        75%         224.000000
        max         255.000000
        dtype: float64
    
    """
    s = pd.Series(im.flatten())
    return s.describe()

def find_edge_gradient(im, threshold = 32, kernel = "sobel"):
    
    """
       把圖片的邊界線描繪出來
    """
    
    assert kernel in ["sobel", "prewitt"], "kernel can be sobel or prewitt."
    
    print type(im)
    im_data = array(im.convert("L"))
    #im_data = im.fromarray(k)
    imx = zeros(im_data.shape)
    imy = zeros(im_data.shape)
    
    if kernel == "prewitt":
        filters.prewitt(im_data, 1, imx)
        filters.prewitt(im_data, 0, imy)
    else:
        filters.sobel(im_data, 1, imx)
        filters.sobel(im_data, 0, imy)

    mag_arr = np.floor(sqrt(imx**2 + imy**2))

    mag_arr = np.floor(255 * mag_arr / np.max(mag_arr))
    arr, cdf = histeq(255 - mag_arr)
    arr = np.floor(arr)
    
    indx, indy = np.where(arr > threshold)
    
    arrc = arr.copy()
    arrc[indx, indy] = 255
    
    return Image.fromarray(arrc.astype(np.uint8)), imx, imy

def find_edge_canny(im, min_pixel_value = 50, max_pixel_value = 100):
    """
        把圖片的邊界線描繪出來
    """
    
    im_data = array(im.convert("L"))
    # im_data, _ = histeq(im_data)
    
    return Image.fromarray(255 - cv2.Canny(im_data.astype(np.uint8), min_pixel_value, max_pixel_value))