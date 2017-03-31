#-*- coding: utf-8 -*-
#-*- coding: cp950 -*-

import matplotlib.pyplot as plt
import argparse
import sys, os
import numpy as np
try:
    import cPickle
except:
    import pickle

def annotate_img_coords(img, n_points):
    fig, axes = plt.subplots()
    axes.imshow(img)
    coords = fig.ginput(n = n_points, timeout = 0)
    plt.close(fig)
    return np.floor(coords).astype(np.int64)

def procedure(parser):
    args = parser.parse_args()
    src = args.src
    output_file = args.output
    append = args.append
    n_points = args.n_points
    if isinstance(output_file, str):
        if append:
            output_file = open(os.path.abspath(output_file), "a")
        else:
            output_file = open(os.path.abspath(output_file), "w")
    for rel_path, dirs, files in os.walk(os.path.abspath(src)):
        for file_name in files:
            file_path = os.path.join(os.path.abspath(rel_path), file_name)
            if file_name[0] == ".":
                continue
            print "[Info] Processing {}".format(file_path)
            img = plt.imread(os.path.join(os.path.abspath(rel_path),file_name))
            coords = annotate_img_coords(img, n_points)
            coords = [",".join(coord.astype(str)) for coord in coords]
            output_file.write(file_name + ":" + " ".join(coords) + "\n")

def main():
    parser = argparse.ArgumentParser(description = "Image annotation tool")
    parser.add_argument("-s", "--source", dest = 'src', metavar = "source",
                        required = True,
                        help = "source directory")
    parser.add_argument("-o", "--output", default = sys.stdout,
                        dest = "output", metavar = "output_file",
                        help = "output file (default: stdout)")
    parser.add_argument("-a", "--append", action = "store_true",
                        dest = "append", help = "append to output file.")
    parser.add_argument("-n", "--n_points", metavar = "n_points", default = 30, type = int,
                        dest = "n_points", help = "number of points for annotation")
    procedure(parser)

if __name__ == "__main__":
    main()




