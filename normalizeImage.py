# -*- coding:utf-8 -*-

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image for normalized")
ap.add_argument("-t", "--type", type=int, default=0, help="normalized type, 0: normal, 1: histogramEqualized")
args = vars(ap.parse_args())

image_file = args["image"]
normalized = args["type"]

image = cv2.imread(image_file)
if type == 0:
    n_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)[:,:,0]
else:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    n_image = cv2.equalizeHist(gray)

filename = image_file[image_file.rfind("/") + 1:]
conponet = filename.split(".")
print("filename:{}".format(conponet[0]))
print("filename:{}".format(conponet[1]))
n_filename = "traffic_sign_image/" + conponet[0] + "_n." + conponet[1]
cv2.imwrite(n_filename, n_image)

cv2.imshow("normalized image", n_image)
cv2.waitKey(0)


