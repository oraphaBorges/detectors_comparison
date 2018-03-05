import cv2
import sys
import sqlite3
import numpy as np
import geometry as gmt
from scipy import stats
from time import time, strftime
from matplotlib import pyplot as plt

SAVE_ONLY = True
NUM_OF_PAIRS = 1
TABLE_NAME = "datas_{}".format(strftime("%y%m%d_%H%M%S"))


def main():
    executeTimeI = time()
    conn = sqlite3.connect("banco.db")
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
            pathImg2 TEXT,
            phase INTEGER
      );""".format(TABLE_NAME)
    )

    # Initiate detectors
    ORB = cv2.ORB.create()
    AKAZE = cv2.AKAZE.create()
    BRISK = cv2.BRISK.create()
    SIFT = cv2.xfeatures2d.SIFT_create()
    SURF = cv2.xfeatures2d.SURF_create()

    methods = {
        "ORB": ORB,
        "AKAZE": AKAZE,
        "BRISK": BRISK,
        "SIFT": SIFT,
        "SURF": SURF
    }

    cases = [
        "Same Object, Same Scale",
        "Same Object, Different Scale",
        "Different Object, Same Scale",
        "Different Object, Different Scale"
    ]

    for case in cases:
        print(case)
        for pair in range(NUM_OF_PAIRS):
            print("Pair {}/{}".format(pair + 1, NUM_OF_PAIRS))
            img1 = cv2.imread("photos/{}/{}a.jpg".format(case, pair), 0)
            img2 = cv2.imread("photos/{}/{}b.jpg".format(case, pair), 0)
            for name, method in methods.items():
                print(name)
                print("Phase One: Compares unaltered images")
                angles_dif, scales, kp1, kp2, origin_matches, origin_values = prep_values(
                    img1, img2, method, name, case, pair)
                origin_values.append(1)

                result = cv2.drawMatches(
                    img1, kp1, img2, kp2, origin_matches, outImg=None)

                save(conn, cursor, tuple(origin_values))
                plot_matches(result, SAVE_ONLY,
                             "results/{}/{}_{}_p1.png".format(case, pair, name))

                print("Phase two: Calculates the transformation")
                angles_mean = origin_values[4]
                scale_mean = origin_values[6]
                dst = gmt.affine_trans(img1, angles_mean, scale_mean)

                _, _, kp1, kp2, matches, values = prep_values(
                    dst, img2, method, name, case, pair)
                values.append(2)

                result = cv2.drawMatches(
                    dst, kp1, img2, kp2, matches, outImg=None)

                save(conn, cursor, tuple(values))
                plot_matches(result, SAVE_ONLY,
                             "results/{}/{}_{}_p2.png".format(case, pair, name))

                print("Phase three: Removes fake matches")
                angles_mean = origin_values[4]
                angles_std = origin_values[5]
                scale_mean = origin_values[6]
                scale_std = origin_values[7]

                angles_dif, scales = gmt.remove_fake_matches(
                    origin_matches, angles_dif, angles_mean, angles_std, scales, scale_mean, scale_std)

                angles_mean = stats.tmean(angles_dif)
                angles_std = stats.tstd(angles_dif)
                scale_mean = stats.tmean(scales)
                scale_std = stats.tstd(scales)

                dst = gmt.affine_trans(img1, angles_mean, scale_mean)
                _, _, kp1, kp2, matches, values = prep_values(
                    dst, img2, method, name, case, pair)
                values.append(3)

                result = cv2.drawMatches(
                    dst, kp1, img2, kp2, matches, outImg=None)

                save(conn, cursor, tuple(values))
                plot_matches(result, SAVE_ONLY,
                             "results/{}/{}_{}_p3.png".format(case, pair, name))

            del img1
            del img2
    conn.close()
    executeTimeF = time()
    print("Test executed in {} seconds".format(executeTimeF - executeTimeI))


def getStats(method, img1, img2):
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
    # matches = sorted(matches, key=lambda x: x.distance)

    return [kp1, kp2, matches, timeF - timeI]


def save(conn, cursor, values):
    cursor.execute(
        """INSERT INTO {} (kp1, kp2, matches, time, anglesMean, anglesSD, scaleMean, scaleSD, technique, situation, pathImg1, pathImg2, phase)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""".format(TABLE_NAME), values)
    conn.commit()


def plot_matches(img, option, path):
    if option == SAVE_ONLY:
        cv2.imwrite(path, img)
    else:
        plt.imshow(img)
        plt.show()


def prep_values(img1, img2, method, name, case, pair):
    values = getStats(method, img1, img2)
    kp1, kp2, matches = values[0], values[1], values[2]
    values[0], values[1], values[2] = len(kp1), len(kp2), len(matches)

    angles_img1 = gmt.g_find_kp_angles(img1, kp1)
    angles_img2 = gmt.g_find_kp_angles(img2, kp2)
    angles_dif = gmt.angles_dif(angles_img1, angles_img2, matches)
    scales = gmt.find_scale_ratios(img1, kp1, img2, kp2, matches)

    angles_mean = stats.tmean(angles_dif)
    angles_std = stats.tstd(angles_dif)

    scale_mean = stats.tmean(scales)
    scale_std = stats.tstd(scales)

    values.append(angles_mean)
    values.append(angles_std)
    values.append(scale_mean)
    values.append(scale_std)
    values.append(name)
    values.append(case)
    values.append("{}a.jpg".format(pair))
    values.append("{}b.jpg".format(pair))

    return angles_dif, scales, kp1, kp2, matches, values


if(__name__ == "__main__"):
    main()
