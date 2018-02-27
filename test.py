import cv2
import sys
import sqlite3
import numpy as np
import geometry as gmt
from scipy import stats
from time import time, strftime
from matplotlib import pyplot as plt

img1 = cv2.imread('photos/Same Object, Same Scale/0a.jpg',0)
img2 = cv2.imread('photos/Same Object, Same Scale/0b.jpg',0)

center2 = gmt.image_center(img2)
m = cv2.getRotationMatrix2D((center2.x, center2.y), 30, 1.2)
dst = cv2.warpAffine(img2, m, (img2.shape[1],img2.shape[0]))


fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(img1)
fig.add_subplot(1,2,2)
plt.imshow(dst)
plt.show()