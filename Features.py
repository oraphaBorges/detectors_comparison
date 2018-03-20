import cv2
import pdb
import sys
import sqlite3
import numpy as np
import geometry as gmt
from time import time, strftime
from matplotlib import pyplot as plt

ARR_LEN = 10
NUM_OF_PAIRS = 1
TABLE_NAME = 'stats_{}'.format(strftime('%y%m%d_%H%M%S'))


def main():
    executeTimeI = time()
    conn = sqlite3.connect('banco.db')
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
        'ORB': ORB,
        # 'AKAZE': AKAZE,
        # 'BRISK': BRISK,
        # 'SIFT': SIFT,
        # 'SURF': SURF
    }

    cases = [
        'Same Object, Same Scale',
        # 'Same Object, Different Scale',
        # 'Different Object, Same Scale',
        # 'Different Object, Different Scale'
    ]

    for case in cases:
        print(case)
        for pair in range(NUM_OF_PAIRS):
            print('Pair {}/{}'.format(pair + 1, NUM_OF_PAIRS))
            img1 = cv2.imread('photos/{}/{}a.jpg'.format(case, pair), 0)
            img2 = cv2.imread('photos/{}/{}b.jpg'.format(case, pair), 0)
            for name, method in methods.items():
                print('\n', name)

                _, matches, kps1, kps2 = process_pair(method, img1, img2)
                print('--------------------')
                stats, _, _, _ = process_pair(method, img1, img2, matches, kps1, kps2)
                stats.append(name)
                stats.append(case)
                stats.append('{}a.jpg'.format(pair))
                stats.append('{}b.jpg'.format(pair))
                stats.append(1)

                save_stats(conn, cursor, stats)

                # cv2.imwrite(path, img)
                # pdb.set_trace()

            del img1
            del img2
    conn.close()
    executeTimeF = time()
    print('Test executed in {} seconds'.format(executeTimeF - executeTimeI))


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

def print_matches(matches,kps1,kps2):
    for match in matches:
        q = kps1[match.queryIdx].pt
        t = kps2[match.trainIdx].pt
        print('({}, {})'.format(q[0], q[1]), '({}, {})'.format(t[0], t[1]))

def process_pair(method, img1, img2, matches_origin=None, kps1_origin=None, kps2_origin=None):
    stats = None
    img_prefix = None
    if matches_origin is None or kps1_origin is None or kps2_origin is None:
        img_prefix = '0'
        stats = get_stats(method, img1, img2)
        kps1_origin, kps2_origin, matches_origin = stats[0], stats[1], stats[2]
        stats[0], stats[1], stats[2] = len(kps1_origin), len(kps2_origin), len(matches_origin)
    else:
        img_prefix = '1'
        stats = []
        stats.append(len(kps1_origin))
        stats.append(len(kps2_origin))
        stats.append(len(matches_origin))
        stats.append(0.0)

    center1 = gmt.kps_center(kps1_origin)
    center2 = gmt.kps_center(kps2_origin)
    print('center1: {}'.format(center1))
    print('center2: {}'.format(center2))

    print('matches_origin:')
    print_matches(matches_origin[:ARR_LEN],kps1_origin,kps2_origin)

    # keypoints distance step
    kps_dist1_origin = gmt.find_kps_dist(center1, kps1_origin)
    kps_dist2_origin = gmt.find_kps_dist(center2, kps2_origin)
    # print('kps_dist1_origin: {}'.format(kps_dist1_origin[:ARR_LEN]))
    # print('kps_dist2_origin: {}'.format(kps_dist2_origin[:ARR_LEN]))

    print('\nkeypoints distance step')
    kps_dist_ratio, kps_dist_ratio_mean, kps_dist_ratio_std = process_kps_feat(
        matches_origin, kps_dist1_origin, kps_dist2_origin, lambda a, b: a / b)
    print('kps_dist_ratio_mean: {}'.format(kps_dist_ratio_mean))
    print('kps_dist_ratio_std: {}\n'.format(kps_dist_ratio_std))
    # print('kps_dist_ratio: {}'.format(kps_dist_ratio[:ARR_LEN]))

    # keypoints angle step
    print('\nkeypoints angle step')
    kps_angles1_origin = gmt.find_kp_angles(center1, kps1_origin)
    kps_angles2_origin = gmt.find_kp_angles(center2, kps2_origin)
    # print('kps_angles1_origin: {}'.format(kps_angles1_origin[:ARR_LEN]))
    # print('kps_angles2_origin: {}'.format(kps_angles2_origin[:ARR_LEN]))

    kps_angles1_mean = np.mean(kps_angles1_origin)
    kps_angles2_mean = np.mean(kps_angles2_origin)
    angles_mean_diff = kps_angles2_mean - kps_angles1_mean
    kps_angles2_origin = np.subtract(kps_angles2_origin, angles_mean_diff)
    # print('kps_angles1_mean: {}'.format(kps_angles1_mean))
    # print('kps_angles2_mean: {}'.format(kps_angles2_mean))
    # print('angles_mean_diff: {}'.format(angles_mean_diff))
    # print('subtracted kps_angles2_origin: {}'.format(kps_angles2_origin[:ARR_LEN]))

    kps_angles_diff, kps_angles_diff_mean, kps_angles_diff_std = process_kps_feat(
        matches_origin, kps_angles1_origin, kps_angles2_origin, lambda a, b: a - b)
    print('kps_angles_diff_mean: {}'.format(kps_angles_diff_mean))
    print('kps_angles_diff_std: {}\n'.format(kps_angles_diff_std))
    # print('kps_angles_diff: {}'.format(kps_angles_diff[:ARR_LEN]))

    # remove outliers step 1/2 std
    matches, kps1, kps2, kps_dist1, kps_dist2,removed_matches = remove_fake_matches(
        matches_origin, kps1_origin, kps2_origin, kps_dist1_origin, kps_dist2_origin, kps_dist_ratio,
        kps_dist_ratio_mean - kps_dist_ratio_std/2, kps_dist_ratio_mean + kps_dist_ratio_std/2)
    # print('final kps_dist1: {}'.format(kps_dist1[:ARR_LEN]))
    # print('final kps_dist2: {}'.format(kps_dist2[:ARR_LEN]))

    print('\npercentage of remaining keypoints (distance 1/2 std): {}'.format(len(matches)/len(matches_origin)))
    print('removed matches 1:')
    print_matches(removed_matches[:ARR_LEN],kps1_origin,kps2_origin)

    img_matches = cv2.drawMatches(img1,kps1,img2,kps2,matches[:ARR_LEN],outImg=None, matchColor = (0,255,0), singlePointColor = (0,0,255))
    cv2.imwrite('results/matches/{}1.jpg'.format(img_prefix), img_matches)

    matches, kps1, kps2, kps_angles1, kps_angles2, removed_matches = remove_fake_matches(
        matches, kps1, kps2, kps_angles1_origin, kps_angles2_origin, kps_angles_diff,
        kps_angles_diff_mean - kps_angles_diff_std/2, kps_angles_diff_mean + kps_angles_diff_std/2)
    # print('final kps_angles1: {}'.format(kps_angles1[:ARR_LEN]))
    # print('final kps_angles2: {}'.format(kps_angles2[:ARR_LEN]))

    print('\nfinal percentage of remaining keypoints (1/2 std): {}'.format(len(matches)/len(matches_origin)))
    print('removed matches 2:')
    print_matches(removed_matches[:ARR_LEN],kps1_origin,kps2_origin)

    img_matches = cv2.drawMatches(img1,kps1,img2,kps2,matches[:ARR_LEN],outImg=None, matchColor = (0,255,0), singlePointColor = (0,0,255))
    cv2.imwrite('results/matches/{}2.jpg'.format(img_prefix), img_matches)

    # remove outliers step 1 std
    matches, kps1, kps2, kps_dist1, kps_dist2, removed_matches = remove_fake_matches(
        matches_origin, kps1_origin, kps2_origin, kps_dist1_origin, kps_dist2_origin, kps_dist_ratio,
        kps_dist_ratio_mean - kps_dist_ratio_std, kps_dist_ratio_mean + kps_dist_ratio_std)

    print('\npercentage of remaining keypoints (distance 1 std): {}'.format(len(matches)/len(matches_origin)))
    print('removed matches 3:')
    print_matches(removed_matches[:ARR_LEN],kps1_origin,kps2_origin)

    img_matches = cv2.drawMatches(img1,kps1,img2,kps2,matches[:ARR_LEN],outImg=None, matchColor = (0,255,0), singlePointColor = (0,0,255))
    cv2.imwrite('results/matches/{}3.jpg'.format(img_prefix), img_matches)

    matches, kps1, kps2, kps_angles1, kps_angles2, removed_matches = remove_fake_matches(
        matches, kps1, kps2, kps_angles1_origin, kps_angles2_origin, kps_angles_diff,
        kps_angles_diff_mean - kps_angles_diff_std, kps_angles_diff_mean + kps_angles_diff_std)
    # print('final kps_angles1: {}'.format(kps_angles1[:ARR_LEN]))
    # print('final kps_angles2: {}'.format(kps_angles2[:ARR_LEN]))

    print('\nfinal percentage of remaining keypoints (1 std): {}'.format(len(matches)/len(matches_origin)))
    print('removed matches 4:')
    print_matches(removed_matches[:ARR_LEN],kps1_origin,kps2_origin)

    img_matches = cv2.drawMatches(img1,kps1,img2,kps2,matches[:ARR_LEN],outImg=None, matchColor = (0,255,0), singlePointColor = (0,0,255))
    cv2.imwrite('results/matches/{}4.jpg'.format(img_prefix), img_matches)

    # final stats recalculation to save on stats list
    kps_dist_ratio, kps_dist_ratio_mean, kps_dist_ratio_std = process_kps_feat(
        matches, kps_dist1, kps_dist2, lambda a, b: a / b)

    kps_angles_diff, kps_angles_diff_mean, kps_angles_diff_std = process_kps_feat(
        matches, kps_angles1, kps_angles2, lambda a, b: a - b)

    stats.append(kps_angles_diff_mean)
    stats.append(kps_angles_diff_std)
    stats.append(kps_dist_ratio_mean)
    stats.append(kps_dist_ratio_std)

    return stats, matches, kps1, kps2


# Applies fn to all the pairs of matches
def kps_fn(matches, kps1, kps2, fn):
    result = np.empty(shape=len(matches))
    i = 0
    for match in matches:
        # result list preserves the ordering from matches list
        q = kps1[match.queryIdx]
        t = kps2[match.trainIdx]
        result[i] = fn(q, t)
        i += 1
    return result


def process_kps_feat(matches, kps_feat1, kps_feat2, fn):
    processed_kps = kps_fn(
        matches, kps_feat1, kps_feat2, fn)
    processed_kps_mean = np.mean(processed_kps)
    processed_kps_std = np.std(processed_kps)
    return processed_kps, processed_kps_mean, processed_kps_std


# Returns two np.arrays with the contents of
# kps_feat1 and kps_feat2, but filtered by
# lower_lim and upper_lim.
# kps_feat1 and kps_feat2 are not arrays of
# keypoints, but arrays of some feature
# or statistic about them, like distance
# from center or angle from horizon.
def remove_fake_matches(matches, kps1, kps2, kps_feat1, kps_feat2, kps_diff, lower_lim, upper_lim):
    j = 0
    new_kps1 = []
    new_kps2 = []
    new_matches = []
    new_kps_feat1 = []
    new_kps_feat2 = []
    removed_matches = []
    for i in range(len(matches)):
        if kps_diff[i] >= lower_lim and kps_diff[i] <= upper_lim:
            # THESE FOUR LINES BELOW MUST BE RUN BEFORE THE LAST TWO
            # BECAUSE kps[1, 2] and kps_feat[1, 2] WERE NOT REORDERED
            # TO FOLLOW matches INDEXING
            new_kps1.append(kps1[matches[i].queryIdx])
            new_kps2.append(kps2[matches[i].trainIdx])
            new_kps_feat1.append(kps_feat1[matches[i].queryIdx])
            new_kps_feat2.append(kps_feat2[matches[i].trainIdx])
            # DO NOT MOVE UP THE TWO LINES BELOW
            matches[i].queryIdx = matches[i].trainIdx = j
            new_matches.append(matches[i])
            j += 1
        else:
            removed_matches.append(matches[i])
    return np.array(new_matches), np.array(new_kps1), np.array(new_kps2), np.array(new_kps_feat1), np.array(new_kps_feat2), np.array(removed_matches)


def save_stats(conn, cursor, stats):
    cursor.execute(
        """INSERT INTO {} (kps1, kps2, matches, time, anglesDiffMean, anglesDiffStd, kpsRatioMean, kpsRatioStd, technique, situation, pathImg1, pathImg2, phase)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""".format(TABLE_NAME), stats)
    conn.commit()


if(__name__ == '__main__'):
    main()
