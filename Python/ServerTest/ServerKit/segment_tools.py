#-*- coding: utf-8 -*-
#-*- coding: cp950 -*-

import cv2
import numpy as np
from collections import defaultdict
from PIL import Image

class NotDetectedError(Exception): pass

def faces_positions(img, num_neighbors = 10, degree_step = -5):
    """
    find_face: Find the face(s) coordinates in `img` and `rotate_degree` (if needed)
        
    Input:
    img::Image -> An Image object.
    num_neighbors::int -> number of neighbors used in face detection.
    degree_step::int -> search step of rotation angle.
    Output:
    coors::numpy.ndarray -> coordinates used for locating face in `img`
    rotate_degree::float -> the degree must `img` rotates in order to find a face.
        
    Note: `find_face` will return (None, None) if no face can be detected.
    """
    
    assert degree_step < 0, "degree_step must be negative."
    
    gray_img = np.array(img.convert("L"))
    face_detector = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")
    faces = face_detector.detectMultiScale(gray_img,
                                           scaleFactor = 1.1,
                                           minNeighbors = num_neighbors,
                                           minSize = (20, 20),
                                           flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    if isinstance(faces, tuple):
        degree = 90
        while isinstance(faces, tuple) and degree >= -90:
            gray_img = np.array(img.rotate(degree).convert("L"))
            faces = face_detector.detectMultiScale(gray_img,
                                                    scaleFactor = 1.1,
                                                    minNeighbors = num_neighbors,
                                                    minSize = (20, 20),
                                                    flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
            degree += degree_step
        else:
            degree -= degree_step
    else:
        degree = 0
                                                                           
    if isinstance(faces, tuple):
        raise NotDetectedError("Can't find face.")
    
    return faces, degree


def crop_img(img, coor, offset = False):
    """
    Crop `img` according to coor.
    Add offset if `offset` is True.
    """
    
    x, y, w, h = coor
    if offset:
        return img.crop((x, y - h/6, x + w, y + 7*h/6))
    else:
        return img.crop((x, y, x+w, y+h))


def mask_img(img, coors, offset = False, color = None):
    """
    Mask `img` according to coordinates in `coors` with color `color`.
    Offset the coordinates by h/6 both top and bottom.
    """
    
    data = np.array(img)
    num_layers = len(data.shape)
    assert num_layers in [1, 3], "Only support RGB or gray image."
    if color is None:
        color = (0, 0, 0)
    if num_layers == 1:
        color = color[0]
    
    for coor in coors:
        x, y, w, h = coor
        if num_layers == 3:
            for ind in range(num_layers):
                if offset:
                    i_start = max(y-h/6, 0)
                    i_end = min(y + 7*h/6, data.shape[0])
                    data[i_start:i_end, x:x+w, ind] = color[ind]
                else:
                    data[y:y+h, x:x+w, ind] = color[ind]
        else:
            if offset:
                i_start = max(y-h/6, 0)
                i_end = min(y + 7*h/6, data.shape[0])
                data[i_start:i_end, :] = color
            else:
                data[y:y+h, x:x+w] = color
    return Image.fromarray(data)

def eyes_positions(img, num_neighbors = 10, degree_step = -5):
    """
    eyes_positions: Find the positions of eyes in the image.
        
    Inputs:
    img::Image -> An `Image` object.
    num_neighbors::int -> number of neighbors used in face detection.
    degree_step::int -> search step of rotation angle.
    Outputs:
    A (`numpy.ndarray`, `int`) or (None, None) -> arrray of the coordinates of eyes in `img` or None if
    nothing can be detected.
    """
    
    assert degree_step < 0, "degree_step must be negative."
    gray_img = np.array(img.convert("L"))
    eyes_detector = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml")
    eyes = eyes_detector.detectMultiScale(gray_img,
                                          scaleFactor = 1.1,
                                          minNeighbors = num_neighbors,
                                          minSize = (20, 20),
                                          flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

    if isinstance(eyes, tuple):
        degree = 90
        while degree >= -90 and len(eyes) < 2:
            gray_img = np.array(img.rotate(degree).convert("L"))
            eyes = eyes_detector.detectMultiScale(gray_img,
                                                  scaleFactor = 1.3,
                                                  minNeighbors = num_neighbors,
                                                  minSize = (20, 20),
                                                  flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
            degree += degree_step
        else:
            degree -= degree_step
    else:
        degree = 0
    
    if len(eyes) < 2:
        raise NotDetectedError("Can't find eyes.")
    
    eyes.view("int32,int32,int32,int32").sort(order = ["f1", "f0"], axis = 0)
    return eyes[0:2, :], degree

def nose_position(img, num_neighbors = 10, degree_step = -5):
    """
    nose_position: Find the positions of nose in the image `img`.
        
    Inputs:
    img::Image -> An `Image` object.
    num_neighbors::int -> number of neighbors used in face detection.
    degree_step::int -> search step of rotation angle.
    Outputs:
    A `numpy.ndarray` or (None, None) -> arrray of the coordinates of nose in `img` or None if
    nothing can be detected.
    """
    
    assert degree_step < 0, "degree_step must be negative."
    
    gray_img = np.array(img.convert("L"))
    nose_detector = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml")
    nose = nose_detector.detectMultiScale(gray_img,
                                          scaleFactor = 1.3,
                                          minNeighbors = num_neighbors,
                                          minSize = (50, 50),
                                          flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

    if isinstance(nose, tuple):
        degree = 90
        while isinstance(nose, tuple) and degree >= -90:
            gray_img = np.array(img.rotate(degree).convert("L"))
            nose = nose_detector.detectMultiScale(gray_img,
                                                  scaleFactor = 1.1,
                                                  minNeighbors = num_neighbors,
                                                  minSize = (50, 50),
                                                  flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
            degree += degree_step
        else:
            degree -= degree_step
    else:
        degree = 0
    
    if isinstance(nose, tuple):
        raise NotDetectedError("Can't find nose.")
    
    nose.view("int32,int32,int32,int32").sort(order = ["f2", "f1"], axis=0)
    return nose[0], degree

def mouth_position(img, num_neighbors = 10, degree_step = -5):
    assert degree_step < 0, "degree_step must be negative."
    
    gray_img = np.array(img.convert("L"))
    mouth_detector = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml")
    mouth = mouth_detector.detectMultiScale(gray_img,
                                            scaleFactor = 1.3,
                                            minNeighbors = num_neighbors,
                                            minSize = (5, 20),
                                            flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    smile_detector = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_smile.xml")
    smile = smile_detector.detectMultiScale(gray_img,
                                            scaleFactor = 1.3,
                                            minNeighbors = num_neighbors,
                                            minSize = (5, 20),
                                            flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

    smile_detector = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_smile.xml")
    smile = smile_detector.detectMultiScale(gray_img,
                                            scaleFactor = 1.3,
                                            minNeighbors = num_neighbors,
                                            minSize = (5, 20),
                                            flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    if isinstance(mouth, tuple) and isinstance(smile, tuple):
        degree = 90
        while isinstance(smile, tuple) and isinstance(mouth, tuple) and degree >= -90:
            gray_img = np.array(img.rotate(degree).convert("L"))
            mouth = mouth_detector.detectMultiScale(gray_img,
                                                    scaleFactor = 1.3,
                                                    minNeighbors = num_neighbors,
                                                    minSize = (5, 20),
                                                    flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
            smile = mouth_detector.detectMultiScale(gray_img,
                                                    scaleFactor = 1.3,
                                                    minNeighbors = num_neighbors,
                                                    minSize = (5, 20),
                                                    flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
            degree += degree_step
        else:
            degree -= degree_step
    else:
        degree = 0

    if isinstance(mouth, tuple) and isinstance(smile, tuple):
        raise NotDetectedError("Can't find mouth")
    elif isinstance(mouth, tuple):
        print "[Info] Smile found."
        result = smile
    elif isinstance(smile, tuple):
        print "[Info] Mouth found"
        result = mouth
    else:
        print "[Info] Both mouth and smile are found"
        result = mouth

    result.view("int32,int32,int32,int32").sort(order = ["f1"], axis = 0)

    return result[-1], degree

def find_all_positions(img):
    result = defaultdict(dict)
    
    # Try finding eyes
    eyes, degree = eyes_positions(img)
    eyes.view("int32,int32,int32,int32").sort(order = ["f0"], axis = 0)
    result["eyes"]["coor"] = eyes
    result["eyes"]["degree"] = degree
    
    # Try finding nose
    nose, degree = nose_position(img)
    result["nose"]["coor"] = nose
    result["nose"]["degree"] = degree
    
    # Try finding eyes
    mouth, degree = mouth_position(img)
    result["mouth"]["coor"] = mouth
    result["mouth"]["degree"] = degree
    
    return result
