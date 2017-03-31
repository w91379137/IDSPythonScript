#-*- coding: utf-8 -*-
#-*- coding: cp950 -*-

import cv2
from segment_tools import *
import numpy as np
import scipy as sp
from scipy import linalg
from PIL import Image

class UnSufficientCoors(Exception): pass
class UnMatchCoors(Exception): pass

def get_angle_horizon(ref_point, point):
    """
        Get the angle between horizontal line and the line between `ref_point` and `po$
        
        Input:
        ref_point::tuple or tuple-like object.
        point::tuple or tuple-like object.
        
        Return:
        degree between the horizontal line and the line passing `ref_point` and `por$
    """
    
    cx, cy = ref_point
    x, y = point
    dx = x - cx
    dy = y - cy
    sin_theta = dy / np.sqrt(dx**2 + dy**2)
    angle_arc = np.arcsin(sin_theta)
    return 180 * angle_arc / np.pi

def annotate_points(img, coors, color = (255, 0, 0), size = 5):
    """
        將圖片上面 標註上 座標點
        
        Annotate points on `img` according to the coordinates in `coors`
        
        Return:
        An annotated image object.
    """
    
    data = np.array(img)
    for x, y in coors:
        cv2.circle(data, (x, y), size, color, -1)
    return Image.fromarray(data)

def eyes_horizon_angle(img):
    degrees = []
    eyes, degree = eyes_positions(img)
    degrees.append(degree)
    eyes.view("i32,i32").sort(order = ["f0"], axis = 0)
    left_eye, right_eye = eyes
    degrees.append(get_angle_horizon(left_eye[0:2], right_eye[0:2]))
    return sum(degrees)

def transition(img, origin, new_origin, resample = Image.CUBIC):
    """
        Make transition to align `origin` to `new_origin`
    """
    
    cx, cy = origin
    nx, ny = new_origin
    a = e = 1
    b = d = 0
    c = cx - nx
    f = cy - ny
    return img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f), resample = resample)

def annotate_face(face_img):
    """
        annotate_face: Annotate face image with its eyes, nose and mouth.
        
        標記出 雙眼 鼻子 嘴巴 位置
    """
    
    # Get mouth coordinates
    mouth, _ = mouth_position(face_img) # Try finding the eyes in ref_img
    mouth_x, mouth_y, w_mouth, h_mouth = mouth
    mouth_coor = (mouth_x + w_mouth/2, mouth_y + h_mouth/2)
    
    # Get nose coordinate
    nose, _ = nose_position(face_img)
    nose_x, nose_y, w_nose, h_nose = nose
    nose_coor = (nose_x + w_nose/2, nose_y + 2*h_nose/3)
    
    # Get eyes
    eyes, _ = eyes_positions(face_img)
    eyes.view("int32,int32,int32,int32").sort(order = ["f0"], axis = 0)
    left_eye, right_eye = eyes
    left_x, left_y, w_left, h_left = left_eye
    right_x, right_y, w_right, h_right = right_eye
    left_coor = (left_x + w_left/2, left_y + h_left/2)
    right_coor = (right_x + w_right/2, right_y + h_right/2)
    
    coors = [left_coor, right_coor, nose_coor, mouth_coor]
    return annotate_points(face_img, coors)

def get_transform(ref_img, target_img):
    ref_face_components = find_all_positions(ref_img)
    target_face_components = find_all_positions(target_img)
    
    #平均 鼻子
    nose_ref_x, nose_ref_y, nose_ref_w, nose_ref_h = ref_face_components["nose"]["coor"]
    nose_target_x, nose_target_y, nose_target_w, nose_target_h = target_face_components["nose"]["coor"]
    nose_ref_coor = [nose_ref_x + nose_ref_w/2, nose_ref_y + 2*nose_ref_h/3]
    nose_target_coor = [nose_target_x + nose_target_w/2, nose_target_y + 2*nose_target_h/3]
    
    #平均 嘴巴
    mouth_ref_x, mouth_ref_y, mouth_ref_w, mouth_ref_h = ref_face_components["mouth"]["coor"]
    mouth_target_x, mouth_target_y, mouth_target_w, mouth_target_h = target_face_components["mouth"]["coor"]
    mouth_ref_coor = [mouth_ref_x + mouth_ref_w/2, mouth_ref_y + 2*mouth_ref_h/3]
    mouth_target_coor = [mouth_target_x + mouth_target_w/2, mouth_target_y + 2*mouth_target_h/3]
    
    eyes_ref = ref_face_components["eyes"]["coor"]
    eyes_target = target_face_components["eyes"]["coor"]
    
    if len(eyes_ref) < 2:
        raise UnSufficientCoors("Unsufficient eyes coordinates in referece image.")
    elif len(eyes_target) < 2:
        raise UnSufficientCoors("Unsufficient eyes coordinates in target image.")
    else:
        if len(eyes_ref) != len(eyes_target):
            raise UnMatchCoors("Eyes coordinates does not match.")
    
    #平均 參考 左右眼
    lefteye_ref, righteye_ref = eyes_ref[0], eyes_ref[1]
    left_ref_x, left_ref_y, left_ref_w, left_ref_h = lefteye_ref
    right_ref_x, right_ref_y, right_ref_w, right_ref_h = righteye_ref
    
    left_ref_coor = [left_ref_x + left_ref_w/2, left_ref_y + left_ref_h/2]
    right_ref_coor = [right_ref_x + right_ref_w/2, right_ref_y + right_ref_h/2]

    #平均 參考 左右眼
    lefteye_target, righteye_target = eyes_target[0], eyes_target[1]
    left_target_x, left_target_y, left_target_w, left_target_h = lefteye_target

    right_target_x, right_target_y, right_target_w, right_target_h = righteye_target
    left_target_coor = [left_target_x + left_target_w/2, left_target_y + left_target_h/2]
    right_target_coor = [right_target_x + right_ref_w/2, right_target_y + right_target_h/2]

    """
        =================================
        [y] = [a][x] + [b]
        
        [y] = [a][x] + [b][1]
        
        [y] = [x 1] [ a ]
        [ ]   [   ] [ b ]
        [ ]   [   ]
        =================================
        [x] = [ a -b ] [x'] + [tx]
        [y]   [ b  a ] [y']   [ty]
        
        [x] = [x' -y'] [a] + [1 0] [tx]
        [y]   [y'  x'] [b]   [0 1] [ty]
        
        [x1] = [ x1' -y1' 1 0 ]  [  a ]
        [y1]   [ y1'  x1' 0 1 ]  [  b ]
        [  ]   [              ]  [ tx ]
        [  ]   [              ]  [ ty ]
        [  ]   [              ]
        [  ]   [              ]
        =================================
    """
    #http://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html
    #放入 y
    y = np.concatenate([left_ref_coor, right_ref_coor, nose_ref_coor, mouth_ref_coor])
    
    A = np.array([[left_target_coor[0], -left_target_coor[1], 1, 0],
                  [left_target_coor[1], left_target_coor[0], 0, 1],
                  [right_target_coor[0], -right_target_coor[1], 1, 0],
                  [right_target_coor[1], right_target_coor[0], 0, 1],
                  [nose_target_coor[0], -nose_target_coor[1], 1, 0],
                  [nose_target_coor[1], nose_target_coor[0], 0, 1],
                  [mouth_target_coor[0], -mouth_target_coor[1], 1, 0],
                  [mouth_target_coor[1], mouth_target_coor[0], 0, 1]])
    
    a, b, tx, ty = linalg.lstsq(A, y)[0]
    M = np.array([[a, -b], [b, a]])
    return M, tx, ty

def face_align(ref_img, target_img):
    M, tx, ty = get_transform(ref_img, target_img)
    a, b, d, e = linalg.inv(M).flatten()
    
    return target_img.transform(ref_img.size, Image.AFFINE, (a, b, -tx, d, e, -ty), resample = Image.CUBIC)