import cv2
import pdb
import sys
import sqlite3
import numpy as np
import geometry as gmt
from time import time, strftime
from matplotlib import pyplot as plt

NUM_OF_PAIRS = 1
TABLE_NAME = "stats_{}".format(strftime("%y%m%d_%H%M%S"))


def main():
    executeTimeI = time()
    conn = sqlite3.connect("banco.db")
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE {} (
            kps1 INTEGER,
            kps2 INTEGER,
            matches INTEGER,
            time FLOAT,
            anglesDiffMean FLOAT,
            anglesDiffStd FLOAT,
            kpsRatioMean FLOAT,
            kpsRatioStd FLOAT,
            technique TEXT,
            situation TEXT,
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
        # "Same Object, Different Scale",
        "Different Object, Same Scale",
        # "Different Object, Different Scale"
    ]

    for case in cases:
        print(case)
        for pair in range(NUM_OF_PAIRS):
            print("Pair {}/{}".format(pair + 1, NUM_OF_PAIRS))
            img1 = cv2.imread("photos/{}/{}a.jpg".format(case, pair), 0)
            img2 = cv2.imread("photos/{}/{}b.jpg".format(case, pair), 0)
            for name, method in methods.items():
                print(name)

                stats = process_pair(method, img1, img2)
                stats.append(name)
                stats.append(case)
                stats.append("{}a.jpg".format(pair))
                stats.append("{}b.jpg".format(pair))
                stats.append(1)

                save_stats(conn, cursor, stats)

                # cv2.imwrite(path, img)
                # pdb.set_trace()

            del img1
            del img2
    conn.close()
    executeTimeF = time()
    print("Test executed in {} seconds".format(executeTimeF - executeTimeI))


def get_stats(method, img1, img2):
    timeI = time()
    # find the keypoints and descriptors with ORB
    kps1, des1 = method.detectAndCompute(img1, None)
    kps2, des2 = method.detectAndCompute(img2, None)

    # create BFMatcher object and
    # match descriptors (query, train)
    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(des1, des2)
    timeF = time()

    return [np.array(kps1), np.array(kps2), np.array(matches), timeF - timeI]


def process_pair(method, img1, img2):
    stats = get_stats(method, img1, img2)
    kps1, kps2, matches = stats[0], stats[1], stats[2]
    stats[0], stats[1], stats[2] = len(kps1), len(kps2), len(matches)

    center1 = gmt.kps_center(kps1)
    center2 = gmt.kps_center(kps2)

    kps_dist1 = gmt.find_kps_dist(center1, kps1)
    kps_dist2 = gmt.find_kps_dist(center2, kps2)

    kps_ratio = gmt.kps_fn(
        kps_dist1, kps_dist2, matches, lambda a, b: a - b)
    kps_ratio_mean = np.mean(kps_ratio)
    kps_ratio_std = np.std(kps_ratio)

    angles1 = gmt.find_kp_angles(img1, center1, kps1)
    angles2 = gmt.find_kp_angles(img2, center2, kps2)

    angles1_mean = np.mean(angles1)
    angles2_mean = np.mean(angles2)
    angles_mean_diff = angles2_mean - angles1_mean
    angles2 = np.subtract(angles2_mean, angles_mean_diff)

    angles_diff = gmt.kps_fn(
        angles1, angles2, matches, lambda a, b: a / b)
    angles_diff_mean = np.mean(angles_diff)
    angles_diff_std = np.std(angles_diff)

    stats.append(angles_diff_mean)
    stats.append(angles_diff_std)
    stats.append(kps_ratio_mean)
    stats.append(kps_ratio_std)

    return stats


def save_stats(conn, cursor, stats):
    cursor.execute(
        """INSERT INTO {} (kps1, kps2, matches, time, anglesDiffMean, anglesDiffStd, kpsRatioMean, kpsRatioStd, technique, situation, pathImg1, pathImg2, phase)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""".format(TABLE_NAME), stats)
    conn.commit()


if(__name__ == "__main__"):
    main()
