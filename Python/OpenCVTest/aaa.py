#-*- coding: utf-8 -*-
#-*- coding: cp950 -*-

from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/Users/w91379137/Desktop/python/OpenCVTest/DYQCGYO.png', 1)

#plt.imshow(img, cmap='gray', interpolation='bicubic')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#http://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#gsc.tab=0

plt.imshow(img2)
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
