import cv2
from scipy import stats
import geometry as gmt
from time import time
import numpy as np
import sqlite3

NUM_OF_PAIRS = 10

def main():
    conn = sqlite3.connect('banco.db')
    cursor = conn.cursor()

    # Initiate detectors
    SIFT = cv2.xfeatures2d.SIFT_create()
    SURF = cv2.xfeatures2d.SURF_create()
    ORB = cv2.ORB.create()
    KAZE = cv2.KAZE.create()
    AKAZE = cv2.AKAZE.create()
    BRISK = cv2.BRISK.create()

    methods = {
        'SIFT': SIFT,
        'SURF': SURF,
        'ORB': ORB,
        'KAZE': KAZE,
        'AKAZE': AKAZE,
        'BRISK': BRISK
    }

    cases = [
      'Same Object and Same Resolution',
      'Same Object and Different Resolution',
      'Different Object and Same Resolution',
      'Different Object and Different Resolution',
      'Same Object and Different Scale',
      'Same Objetct and Different Angle',
      'Image Rotated 45º',
      'Image Rotated 90º'
    ]

    for case in cases:
      for pair in range(NUM_OF_PAIRS):
        for name, method in methods.items():
          print(name)
          print(case)
          values = getStats(cv2.imread('photos/{}/{}a.jpg'.format(case,pair),0), cv2.imread('photos/{}/{}b.jpg'.format(case,pair),0))
          values.append(name)
          values.append(case)
          values.append('{}a.jpg'.format(pair))
          values.append('{}b.jpg'.format(pair))
          cursor.execute("""
            INSERT INTO datas (kp1,kp2,matches,time,anglesMean,anglesSD,distances1Mean,distances1SD,distaces2Mean,distances2SD,technique,situation,pathImg1,pathImg2)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, tuple(values))

    conn.close()


def getStats(img1, img2):
    timeI = time()
    # find the keypoints and descriptors with ORB
    kp1, des1 = method.detectAndCompute(img1, None)
    kp2, des2 = method.detectAndCompute(img2, None)
    timeF = time()

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors. (query,train)
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # matches = np.sort(a,order='distance')

    # Standard Deviation
    center1 = gmt.image_center(img1)
    center2 = gmt.image_center(img2)
    angles = gmt.find_kp_angles(kp1, kp2, matches, center1, center2)
    distances1 = gmt.find_kp_distances(kp1, center1)
    distances2 = gmt.find_kp_distances(kp2, center2)

    return [len(kp1),
            len(kp2), len(matches), timeF - timeI, stats.tmean(angles), stats.tstd(angles), stats.tmean(distances1), stats.tstd(distances1), stats.tmean(distances2), stats.tstd(distances2)]


if(__name__ == '__main__'):
    main()
