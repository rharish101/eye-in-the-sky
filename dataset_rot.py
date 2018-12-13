#!/usr/bin/env python3
"""Generate augmented dataset for Eye-in-the-Sky."""
from __future__ import print_function
import cv2
import numpy as np
import glob
import random
import os
from libtiff import TIFF as t
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pickle

EXCLUDE = ["2.tif", "7.tif", "12.tif"]

random.seed(5)

gnd_path = "./gt/"
img_path = "./sat/"


def get_colours():
    if "colours.pkl" in os.listdir("./"):
        with open("colours.pkl", "rb") as cfile:
            return pickle.load(cfile)
    colours = set()
    for image_path in glob.glob(gnd_path + "*.tif"):
        tiff = t.open(image_path)
        img_gt = tiff.read_image()
        tiff.close()
        colours.update(list(map(tuple, np.reshape(img_gt, (-1, 3)).tolist())))
    colours = sorted(colours)
    with open("colours.pkl", "wb") as cfile:
        pickle.dump(colours, cfile)
    return colours


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Dataset generator for Eye-in-the-Sky",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path", type=str, default="./", help="path to dataset"
    )
    args = parser.parse_args()
    if args.data_path[-1] != "/":
        args.data_path += "/"

    gnd_path = args.data_path + "gt/"
    img_path = args.data_path + "sat/"

    gnd_save_path = gnd_path + "rotated/"
    img_save_path = img_path + "rotated/"

    if not os.path.exists(gnd_save_path):
        os.mkdir(gnd_save_path)
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    colours = get_colours()
    print("Total {} classes found".format(len(colours)))

    count = 0
    for image_path in sorted(glob.glob(img_path + "*.tif")):
        name = image_path.split("/")[-1]
        if name in EXCLUDE:
            continue

        tiff = t.open(image_path)
        img_sat = tiff.read_image()
        tiff.close()

        tiff = t.open(gnd_path + name)
        img_gt = tiff.read_image()
        tiff.close()

        # Thresholding w.r.t. colour
        new_gt = [colours.index(tuple(i)) for i in img_gt.reshape((-1, 3))]
        img_gt = np.reshape(new_gt, (*(img_gt.shape[:2]), 1)).astype(np.uint8)

        for x in range(100, img_sat.shape[0], 100):
            for y in range(100, img_sat.shape[1], 100):
                temp_sat = img_sat[x - 100 : x, y - 100 : y, :]
                temp_gt = img_gt[x - 100 : x, y - 100 : y, :]

                rows, cols, _ = temp_sat.shape

                for angle in (0, 90, 180, 270):
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                    temp_sat_a = cv2.warpAffine(temp_sat, M, (cols, rows))

                    # Border value is set to value of unknown class i.e. white
                    temp_gt_a = cv2.warpAffine(
                        temp_gt,
                        M,
                        (cols, rows),
                        borderValue=colours.index((255, 255, 255)),
                    )

                    # Ensure thresholding is valid
                    temp_gt_a = np.expand_dims(
                        np.clip(temp_gt_a, 0, len(colours) - 1), axis=2
                    )

                    cx_a, cy_a, _ = temp_sat_a.shape
                    cx_a //= 2
                    cy_a //= 2

                    sat_64 = temp_sat_a[
                        cx_a - 32 : cx_a + 32, cy_a - 32 : cy_a + 32, :
                    ]
                    gt_64 = temp_gt_a[
                        cx_a - 32 : cx_a + 32, cy_a - 32 : cy_a + 32, :
                    ]

                    tiff = t.open(
                        gnd_save_path + str(count).zfill(4) + ".tif", mode="w"
                    )
                    tiff.write_image(gt_64, None, True)
                    tiff.close()

                    tiff = t.open(
                        img_save_path + str(count).zfill(4) + ".tif", mode="w"
                    )
                    tiff.write_image(sat_64, None, True)
                    tiff.close()

                    count += 1
                    print(
                        "\rGenerated {} input-output pair(s)".format(count),
                        end="",
                    )
    print("")
