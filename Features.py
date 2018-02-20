import cv2
import sys
import sqlite3
import numpy as np
import geometry as gmt
from scipy import stats
from time import time, strftime

NUM_OF_PAIRS = 3
TABLE_NAME = 'datas_{}'.format(strftime('%y%m%d_%H%M%S'))

def main():
    executeTimeI = time()
    conn = sqlite3.connect('banco.db')
    cursor = conn.cursor()
    cursor.execute(
      """CREATE TABLE {} (
            technique TEXT,
            situation TEXT,
            kp1 INTEGER,
            kp2 INTEGER,
            matches INTEGER,
            time FLOAT,
            anglesMean FLOAT,
            anglesSD FLOAT,
            scaleMean FLOAT,
            scaleSD FLOAT,
            pathImg1 TEXT,
            pathImg2 TEXT
      );""".format(TABLE_NAME)
    )

    # Initiate detectors
    SIFT = cv2.xfeatures2d.SIFT_create()
    SURF = cv2.xfeatures2d.SURF_create()
    ORB = cv2.ORB.create()
    # KAZE = cv2.KAZE.create()
    AKAZE = cv2.AKAZE.create()
    BRISK = cv2.BRISK.create()

    methods = {
        'SIFT': SIFT,
        'SURF': SURF,
        'ORB': ORB,
        # 'KAZE': KAZE,
        'AKAZE': AKAZE,
        'BRISK': BRISK
    }

    cases = [
        'Same Object, Same Scale',
        'Same Object, Different Scale',
        'Different Object, Same Scale',
        'Different Object, Different Scale',
    ]

    for case in cases:
      print(case)
      for pair in range(NUM_OF_PAIRS):
        print('Pair {}/{}'.format(pair, NUM_OF_PAIRS))
        img1 = cv2.imread('photos/{}/{}a.jpg'.format(case,pair),0)
        img2 = cv2.imread('photos/{}/{}b.jpg'.format(case,pair),0)
        for name, method in methods.items():
          print(name)
          try:
            values = getStats(method,img1,img2)
            values.append(name)
            values.append(case)
            values.append('{}a.jpg'.format(pair))
            values.append('{}b.jpg'.format(pair))
            cursor.execute("""
              INSERT INTO {} (kp1,kp2,matches,time,anglesMean,anglesSD,scaleMean,scaleSD,technique,situation,pathImg1,pathImg2)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
              """.format(TABLE_NAME), tuple(values))
            conn.commit()
          except Exception:
            print(sys.exc_info())
            pass
        del img1
        del img2
    conn.close()
    executeTimeF = time()
    print('Test executed in {} minutes'.format(executeTimeF-executeTimeI))


def getStats(method,img1, img2):
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

    # Standard Deviation
    center1 = gmt.image_center(img1)
    center2 = gmt.image_center(img2)
    angles = gmt.find_kp_angles(kp1, kp2, matches, center1, center2)
    scale =  gmt.find_scale(kp1,kp2,matches,center1,center2)

    return [len(kp1),len(kp2), len(matches), timeF - timeI, stats.tmean(angles), stats.tstd(angles), stats.tmean(scale), stats.tstd(scale)]


if(__name__ == '__main__'):
    main()
