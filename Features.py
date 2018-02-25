import cv2
import sys
import sqlite3
import numpy as np
import geometry as gmt
from scipy import stats
from time import time, strftime
from matplotlib import pyplot as plt

NUM_OF_PAIRS = 1
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
    # SIFT = cv2.xfeatures2d.SIFT_create()
    # SURF = cv2.xfeatures2d.SURF_create()
    ORB = cv2.ORB.create()
    # # KAZE = cv2.KAZE.create()
    # AKAZE = cv2.AKAZE.create()
    # BRISK = cv2.BRISK.create()

    methods = {
        # 'SIFT': SIFT,
        # 'SURF': SURF,
        'ORB': ORB,
        # 'KAZE': KAZE,
        # 'AKAZE': AKAZE,
        # 'BRISK': BRISK
    }

    cases = [
        'Same Object, Same Scale',
        # 'Same Object, Different Scale',
        # 'Different Object, Same Scale',
        # 'Different Object, Different Scale',
    ]

    for case in cases:
      print(case)
      for pair in range(NUM_OF_PAIRS):
        print('Pair {}/{}'.format(pair + 1, NUM_OF_PAIRS))
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

            # print("Stats calculated...")
            # mean_angles = values[4]
            # mean_scale = values[6]
            # center2 = gmt.image_center(img2)
            # m = cv2.getRotationMatrix2D((center2.x, center2.y),mean_angles,mean_scale)
            # dst = cv2.warpAffine(img2,m,img2.shape)
            # # plt.imshow(dst)
            # # plt.imshow(img2)
            # plt.imshow(cv2.drawMatchesK(img1,kp1,dst,kp2,[]))
            # plt.show()
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

    angles_img1 = gmt.g_find_kp_angles(img1,kp1)
    angles_img2 = gmt.g_find_kp_angles(img2,kp2)
    dif = gmt.angles_dif(angles_img1,angles_img2,matches)
    scale =  gmt.find_scale_ratios(img1, kp1, img2, kp2, matches)

    mean_angles = stats.tstd(dif)
    std_angles = stats.tstd(dif)

    mean_scale = stats.tmean(scale)
    std_scales = stats.tstd(scale)

    
    print("Stats calculated...")
    center2 = gmt.image_center(img2)
    m = cv2.getRotationMatrix2D((center2.x, center2.y), mean_angles, mean_scale)
    dst = cv2.warpAffine(img2, m, img2.shape)
    plt.imshow(cv2.drawMatches(img1, kp1, dst, kp2, [], None))
    plt.show()

    return [len(kp1),len(kp2), len(matches), timeF - timeI, mean_angles, std_angles, mean_scale, std_scales]

if(__name__ == '__main__'):
    main()
