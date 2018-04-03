import cv2
import pdb
import sys
import numpy as np
import operator as op
import functools as ft
import geometry as gmt
from time import time, strftime
from matplotlib import pyplot as plt

ARR_LEN = 10
NUM_OF_PAIRS = 1
TABLE_NAME = 'stats_{}'.format(strftime('%y%m%d_%H%M%S'))


def main():
    executeTimeI = time()

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
            print(f'Pair {pair + 1}/{NUM_OF_PAIRS}')
            img1 = cv2.imread(f'photos/{case}/{pair}a.jpg', 0)
            img2 = cv2.imread(f'photos/{case}/{pair}b.jpg', 0)
            for name, method in methods.items():
                print(f'\n{name}')

                (matches, kps1, kps2, _) = get_stats(method, img1, img2)

                (matches, kps1, kps2) = process_pair(case, name, pair,1, img1, img2, matches, kps1, kps2)
                print('\n--------------------\n')
                process_pair(case, name, pair,2, img1, img2, matches, kps1, kps2, 1.0)

            del img1
            del img2
    executeTimeF = time()
    print(f'Test executed in {executeTimeF - executeTimeI} seconds')

def process_pair(case, name, pair, iteration, img1, img2, matches_origin, kps1_origin, kps2_origin, std_amount = 1/2,thresh = 0.05):
    matches, kps1, kps2, kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, removed_matches = remove_outliers(
        matches_origin, kps1_origin, kps2_origin, dist_step, std_amount)

    is_below_error = stats_eq(kps_feat_diff,thresh)
    below_error = ft.reduce(lambda x,z: x+1 if z==True else x,is_below_error,0)

    print(f'keypoints below {thresh*100}% error: {below_error} of {len(is_below_error)} ({below_error/len(is_below_error)*100}%)')
    write_matches_img(f'results/matches/{case}_{name}_{pair}_{iteration}_dist.jpg',
        img1, kps1, img2, kps2, matches)
    write_histogram(f'results/matches/{case}_{name}_{pair}_{iteration}_dist.png', kps_feat_diff,kps_feat_diff_mean,kps_feat_diff_std,
        f'{case}, {name}, distance - Pair {pair}, i={iteration}','Distance ratio','Frequency')
    print(f'remaining keypoints with {std_amount} std: {len(matches)/len(matches_origin)} %')
    print('removed matches:')
    print_matches(removed_matches[:ARR_LEN], kps1_origin, kps2_origin)

    matches, kps1, kps2, kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, removed_matches = remove_outliers(
        matches, kps1, kps2, angle_step, std_amount)

    is_below_error = stats_eq(kps_feat_diff,thresh)
    below_error = ft.reduce(lambda x,z: x+1 if z==True else x,is_below_error,0)

    print(f'keypoints below {thresh*100}% error: {below_error} of {len(is_below_error)} ({below_error/len(is_below_error)*100}%)')
    write_matches_img(f'results/matches/{case}_{name}_{pair}_{iteration}_angle.jpg',
        img1, kps1, img2, kps2, matches)
    write_histogram(f'results/matches/{case}_{name}_{pair}_{iteration}_angle.png', kps_feat_diff,kps_feat_diff_mean,kps_feat_diff_std,
        f'{case}, {name}, angle - Pair {pair}, i={iteration}','Angle ratio','Frequency')
    print(f'remaining keypoints with {std_amount} std: {len(matches)/len(matches_origin)} %')
    print('removed matches:')
    print_matches(removed_matches[:ARR_LEN], kps1_origin, kps2_origin)

    return matches, kps1, kps2

# thresh is in percentage
def stats_eq(diffs, thresh=0.05):
    return np.array(list(map(ft.partial(op.ge, 1 + thresh), diffs)))

def write_histogram(path, xs,mean,std, title=None, xlabel=None, ylabel=None):
    frenquency = len(xs)//5
    plt.hist(xs,frenquency, density=1)
    plt.title(title if title is not None else '')
    plt.xlabel(xlabel if xlabel is not None else '')
    plt.ylabel(ylabel if ylabel is not None else '')
    plt.grid(True)

    plt.axvline(mean-std,c='g',ls='--')
    plt.text(mean-std, 0, r'$\sigma^-$')
    plt.axvline(mean,c='r',ls='--')
    plt.text(mean,0, r'$\mu$')
    plt.axvline(mean+std,c='g',ls='--')
    plt.text(mean+std,0, r'$\sigma^+$')

    plt.savefig(path)
    plt.show()

def write_matches_img(path, img1, kps1, img2, kps2, matches):
    img_matches = cv2.drawMatches(
        img1, kps1, img2, kps2, matches[:ARR_LEN],
        outImg = None, matchColor = (0, 255, 0), singlePointColor = (0, 0, 255))
    cv2.imwrite(path, img_matches)

def get_stats(method, img1, img2):
    timeI = time()
    kps1, des1 = method.detectAndCompute(img1, None)
    kps2, des2 = method.detectAndCompute(img2, None)

    # create BFMatcher object and
    # match descriptors (query, train)
    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(des1, des2)
    timeF = time()

    return (np.array(matches), np.array(kps1), np.array(kps2), timeF - timeI)

def print_matches(matches,kps1,kps2):
    for match in matches:
        q = kps1[match.queryIdx].pt
        t = kps2[match.trainIdx].pt
        print(f'({q[0]}, {q[1]})', f'({t[0]}, {t[1]})')

def dist_step(matches, kps1, kps2):
    center1 = gmt.kps_center(kps1)
    center2 = gmt.kps_center(kps2)

    dist1 = gmt.find_kps_dist(center1, kps1)
    dist2 = gmt.find_kps_dist(center2, kps2)

    ratio, ratio_mean, ratio_std = process_kps_feat(
        matches, dist1, dist2, op.truediv)
    return dist1, dist2, ratio, ratio_mean, ratio_std

def angle_step(matches, kps1, kps2):
    center1 = gmt.kps_center(kps1)
    center2 = gmt.kps_center(kps2)

    angles1 = gmt.find_kp_angles(center1, kps1)
    angles2 = gmt.find_kp_angles(center2, kps2)

    angles1_mean = np.mean(angles1)
    angles2_mean = np.mean(angles2)
    angles_mean_diff = angles2_mean - angles1_mean
    angles2 = np.subtract(angles2, angles_mean_diff)

    diff, diff_mean, diff_std = process_kps_feat(
        matches, angles1, angles2, op.truediv)
    return angles1, angles2, diff, diff_mean, diff_std

def remove_outliers(matches_origin, kps1_origin, kps2_origin, step_fn, std_amount):
    print(f'\nremove outliers with {step_fn.__name__}')
    (kps_feat1, kps_feat2, kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std) = step_fn(
        matches_origin, kps1_origin, kps2_origin)
    print(f'mean before: {kps_feat_diff_mean}')
    print(f'std before: {kps_feat_diff_std}')

    matches, kps1, kps2, kps_feat1, kps_feat2, removed_matches = remove_fake_matches(
        matches_origin, kps1_origin, kps2_origin, kps_feat1, kps_feat2, kps_feat_diff,
        kps_feat_diff_mean - (kps_feat_diff_std * std_amount),
        kps_feat_diff_mean + (kps_feat_diff_std * std_amount))

    (kps_feat1, kps_feat2, kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std) = step_fn(
        matches, kps1, kps2)
    print(f'mean after: {kps_feat_diff_mean}')
    print(f'std after: {kps_feat_diff_std}')
    return matches, kps1, kps2, kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, removed_matches

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


if(__name__ == '__main__'):
    main()
