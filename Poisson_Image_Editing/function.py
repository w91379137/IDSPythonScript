# -*- coding: utf-8 -*-

import cv2
import numpy as np

kernel = np.zeros((3,3), np.float32)
kernel[1,0] = 1
kernel[1,2] = 1
kernel[1,1] = -4
kernel[0,1] = 1
kernel[2,1] = 1

def divergence(img):
    img = img.astype(np.float32)
    return cv2.filter2D(img, -1, kernel)

def solve_image_1(img_target, img_div):
    
    w, h = img_div.shape
    A = np.zeros((w * h, w * h), np.float32) #計算用的矩陣
    y = np.zeros((w * h), np.float32)
    
    for idx_h in range(h):
        for idx_w in range(w):
            
            idx_row = idx_w + idx_h * w
            row = np.zeros((w * h), np.float32) #要填入 0 -1 4
            
            if idx_w == 0 or idx_w == w - 1 or idx_h == 0 or idx_h == h - 1:
                #邊界
                y[idx_row] = img_target[idx_w, idx_h]
                row[idx_row] = 1
            else:
                #中間
                y[idx_row] = img_div[idx_w, idx_h]
                row[idx_row - w] = 1
                row[idx_row - 1] = 1
                row[idx_row] = -4
                row[idx_row + 1] = 1
                row[idx_row + w] = 1
                
            A[idx_row, :] = row
        
    #求x
    x = np.linalg.solve(A, y)
    ans = img_target.copy()
    for idx_h in range(h):
        for idx_w in range(w):
            idx_row = idx_w + idx_h * w
            ans[idx_w, idx_h] = min(max(x[idx_row], 0), 255)
    
    return ans.astype(np.uint8)

def auto_mask(img):
    w, h, color = img.shape
    ans = img.copy()
    for idx_h in range(h):
        for idx_w in range(w):
            
            check = 0
            for idx_color in range(color):
                if ans[idx_w, idx_h, idx_color] > 240:
                    check += 1
            
            if check >= color:
                ans[idx_w, idx_h, :] = 0
            else:
                ans[idx_w, idx_h, :] = 255
                
    return ans

def solve_image_2(img_target, img_div, img_mask):
    
    w, h = img_div.shape
    A = np.zeros((w * h, w * h), np.float32) #計算用的矩陣
    y = np.zeros((w * h), np.float32)
    
    for idx_h in range(h):
        for idx_w in range(w):
            
            idx_row = idx_w + idx_h * w
            row = np.zeros((w * h), np.float32) #要填入 0 -1 4
            
            if idx_w == 0 or idx_w == w - 1 or idx_h == 0 or idx_h == h - 1:
                #邊界
                y[idx_row] = img_target[idx_w, idx_h]
                row[idx_row] = 1
            elif img_mask[idx_w, idx_h] == 0:
                y[idx_row] = img_target[idx_w, idx_h]
                row[idx_row] = 1
            else:
                #中間
                y[idx_row] = img_div[idx_w, idx_h]
                row[idx_row - w] = 1
                row[idx_row - 1] = 1
                row[idx_row] = -4
                row[idx_row + 1] = 1
                row[idx_row + w] = 1
                
            A[idx_row, :] = row
        
    #求x
    #_, x = cv2.solve(A, y)
    x = np.linalg.solve(A, y)
    
    ans = img_target.copy()
    for idx_h in range(h):
        for idx_w in range(w):
            idx_row = idx_w + idx_h * w
            ans[idx_w, idx_h] = min(max(x[idx_row], 0), 255)
    
    return ans.astype(np.uint8)