# -*- coding: utf-8 -*-
import numpy as np
import math

def complex2hsv(array):
    """
    輸入一個 2d 複數影像 輸出 hsv
    """
    w, h = array.shape
    array_h = np.full((w, h), 180) + np.angle(array) * 180 / math.pi
    #array_h = np.angle(array)
    array_l = np.full((w, h), 1) - np.full((w, h), 1) / (np.full((w, h), 1) + np.power(np.absolute(array), 0.3))      
    #array_l = np.power(0.5, numpy.absolute(array)) 
    array_s = np.full((w, h), 1)
    
    array_ans = np.zeros((w, h, 3), dtype=np.float32)
    for x in range(0, w):
        for y in range(0, h):
            array_ans[x, y, 0] = array_h[x, y]
            array_ans[x, y, 1] = array_l[x, y]
            array_ans[x, y, 2] = array_s[x, y]
    
    return array_ans