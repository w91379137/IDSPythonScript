#-*- coding: utf-8 -*-
#-*- coding: cp950 -*-

import edge_tools
import segment_tools
import face_align_tools
from PIL import Image
import os, argparse
import numpy as np
import random
import string

class MultiFaceError(Exception): pass # This error is used when multiple faces are detected in reference image.

def random_prefix(length = 5):
    return ''.join([random.choice(string.ascii_lowercase + string.digits + string.ascii_uppercase) for _ in range(length)]) + "_"

def preprocess(args):
    """
        the main function of preprocessing.
        """
    
    # Set up options
    src = args.src
    dest = args.dest
    collect_path = args.collect_path
    formats = args.formats
    ref_img_path = args.ref_img_path
    width = args.width
    debug = args.debug
    if debug:
        print args.__dict__
    # Make necessary directories if there is not.
    if not os.path.exists(dest):
        os.mkdir(dest)
    if not os.path.exists(collect_path):
        os.mkdir(collect_path)

    # Open referce image and trying to find the face in it.
    try:
        ref_img_origin = Image.open(os.path.abspath(ref_img_path))
    except IOError as e:
        print "[IOError] Can't open the reference imgae: {}".format(ref_img_path)
        print "[Info] Terminating...."
        return 1

    face_ref_coor, degree_ref = segment_tools.faces_positions(ref_img_origin)
    
    # Only one face is allowed in referece image. Raise error if it isn't.
    # Crop the origin image to get the face image.
    if face_ref_coor.shape[0] > 1:
        raise MultiFaceError("Detect multiple faces in reference image. There should be only one face.")
    face_ref = segment_tools.crop_img(ref_img_origin, face_ref_coor[0], offset = True)

    # Adjust that image to make eyes lie on horizontal line.
    try:
        eye_angle = face_align_tools.eyes_horizon_angle(face_ref)
    except segment_tools.NotDetectedError:
        print "[NotDetectedError] This reference image is not good enough. The program can't make the eyes horizontal."
        print "[NotDetectedError] Pleas use another reference image."
        print "Terminating...."
        return 1

    total_degree = eye_angle + degree_ref
    img_ref_rotated = ref_img_origin.rotate(total_degree, resample = Image.CUBIC)
    face_ref_coor, _ = segment_tools.faces_positions(img_ref_rotated)
    face_ref = segment_tools.crop_img(img_ref_rotated, face_ref_coor[0], offset = True)
    
    # Resize the reference face to desired witdh (but preserve the width/heigh ratio.)
    ref_width, ref_heigh = face_ref.size
    face_ref = face_ref.resize((width, ref_heigh*width/ref_width))
    if debug:
        face_ref.show()
    
    ref_file_name = os.path.basename(ref_img_path)
    face_ref.save(os.path.join(os.path.abspath(dest), "ref_" + ref_file_name))
    print "[Info] Complete preprocess of reference image."

    # Walk through the source directory.
    print "[Info] Start processing files in {src}.".format(src = os.path.abspath(src))
    for rel_path, dir_names, file_names in os.walk(os.path.abspath(src)):
        for filename in file_names:
            if np.any(map(filename.endswith, formats)):
                file_path = os.path.join(os.path.abspath(rel_path), filename)
                print "[Info] Start processing {file_path}.".format(file_path = file_path)
                try:
                    target_img_origin = Image.open(file_path)
                except IOError as e:
                    print "[IOError] Can not open {}".format(file_path)
                    print "[Info] Passing this image."
                    continue
                
                # Try to find faces in target image. If don't, copy it to collection directory.
                try:
                    faces_target_coors, degree_target = segment_tools.faces_positions(target_img_origin)
                except segment_tools.NotDetectedError as e:
                    print "[NotDetectedError] Does not find any face in {filename}. Collect it into {collect_path}".format(filename = filename, collect_path = collect_path)
                    target_img_origin.save(os.path.join(os.path.abspath(collect_path), filename))
                    continue # Brake loop for not finding any face in the picture.

    # Adjust all found faces to make them just.
    target_img_rotated = target_img_origin.rotate(degree_target, resample = Image.CUBIC)
    for face_coor in faces_target_coors:
        temp_img = segment_tools.crop_img(target_img_rotated, face_coor, offset=True)
        try:
            eyes_degree = face_align_tools.eyes_horizon_angle(temp_img)
        except segment_tools.NotDetectedError:
            eyes_degree = 0
            face_target = temp_img.rotate(eyes_degree)
            temp_file_name = random_prefix() + filename
            if debug:
                face_target.show()
                face_target.save(os.path.join(os.path.abspath(dest), temp_file_name))
                temp_aligned_file_name = "aligned_" + temp_file_name
                try:
                    face_target_aligned = face_align_tools.face_align(face_ref, face_target)
                    face_target_aligned.save(os.path.join(os.path.abspath(dest), temp_aligned_file_name))
                except segment_tools.NotDetectedError:
                    print "[AlignError] Can't align face. Moving to {collection}.".format(collection = collect_path)
                    face_target.save(os.path.join(os.path.abspath(collect_path), "not_aligned_" + temp_file_name))
                    print "[Info] Saving {}".format(os.path.join(os.path.abspath(collect_path), "not_aligned_" + temp_file_name))
                    continue
                    masked_target_img = segment_tools.mask_img(target_img_rotated, faces_target_coors)

                if debug:
                    masked_target_img.show()
                    masked_target_img.save("masked.jpg")
            
                    try:
                        while True:
                            temp_face_coors, temp_degree = segment_tools.faces_positions(masked_target_img)
                            temp_img = masked_target_img.rotate(temp_degree, resample = Image.CUBIC)
                        if debug:
                            print "temp_face_coors", temp_face_coors
                            print "[Info] Multiple faces are found in {file_path}".format(file_path = file_path)
                        for face_coor in temp_face_coors:
                            temp_face = segment_tools.crop_img(temp_img, face_coor, offset = True)
                            eye_angle = face_align_tools.eyes_horizon_angle(temp_face)
                            face_target = temp_face.rotate(eye_angle, resample = Image.CUBIC)
                            if debug:
                                face_target.show()
                                face_target_aligned = face_align_tools.face_align(face_ref, face_target)
                                temp_file_name = random_prefix() + filename
                                temp_aligned_file_name = "aligned_" + temp_file_name
                                print "[Info] Sucessful aligned {}".format(temp_file_name)
                            if debug:
                                masked_target_img.show()
                    except segment_tools.NotDetectedError:
                        file_path = os.path.join(os.path.abspath(rel_path), filename)
                        print "[Info] Complete searching faces in {file_path}".format(file_path = file_path)

def main():
    
    parser = argparse.ArgumentParser(description="Spe3D Images Preprocessing")
    parser.add_argument("-s", "--source", dest='src', metavar = 'source',
                        required = True,
                        help= "source directory")
    parser.add_argument("-d", "--destination", dest='dest', metavar = 'to',
                                            default = 'processed_imgs',
                                            help = "destination directory")
    parser.add_argument("-c", "--collect_path", dest='collect_path', metavar = "collect_path",
                                            default = "failed_imgs",
                                            help = "Path to where to collect pictures which fail the preprocess")
    parser.add_argument("-f", '--formats', dest = "formats",
                                            action = "store",
                                            nargs = '*',
                                            default = ['jpg', 'png', 'jpeg', 'gif'],
                                            metavar = 'format', help = "image formats")
    parser.add_argument("-r", '--ref', dest = 'ref_img_path',
                                            required = True,
                                            metavar = 'reference_image', help = "path of reference image")
    parser.add_argument("-w", '--width', dest = 'width',
                                            default = 300,
                                            type = int,
                                            metavar = 'output width', help = "output image width")
    parser.add_argument("--debug", dest = "debug",
                                            action = "store_true",
                                            help = "Running in debug mode.")
    args = parser.parse_args()
    preprocess(args)

if __name__ == "__main__":
    
    import sys, traceback
    
    try:
        main()
    
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback)
        print(parser.format_help())


