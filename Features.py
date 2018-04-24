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
    ORB = cv2.ORB.create(nfeatures=5000)
    AKAZE = cv2.AKAZE.create()
    BRISK = cv2.BRISK.create()
    SIFT = cv2.xfeatures2d.SIFT_create()
    SURF = cv2.xfeatures2d.SURF_create()

    methods = {
        'ORB': ORB,
        'AKAZE': AKAZE,
        'BRISK': BRISK,
        # 'SIFT': SIFT,
        # 'SURF': SURF
    }

    cases = [
        'Same Object, Same Scale',
        # 'Same Object, Different Scale',
        'Different Object, Same Scale',
        # 'Different Object, Different Scale'
    ]

    for case in cases:
        print('\n\n\n####################')
        print(f'Running case: {case}')
        for pair in range(NUM_OF_PAIRS):
            print('\n\n+++++++++++++++')
            print(f'Pair {pair + 1} of {NUM_OF_PAIRS}')
            img1 = cv2.imread(f'photos/{case}/{pair}a.jpg', 0)
            img2 = cv2.imread(f'photos/{case}/{pair}b.jpg', 0)
            for name, method in methods.items():
                print('\n---------------')
                print(f'Running method: {name}')

                (matches, kps1, kps2, _) = get_stats(method, img1, img2)

                (matches, kps1, kps2) = process_pair(
                    case, name, pair, 1, img1, img2, matches, kps1, kps2)
                print('\n=====\nRunning 2nd iteration')
                process_pair(case, name, pair, 2, img1,
                             img2, matches, kps1, kps2, 1.0)

            del img1
            del img2
    executeTimeF = time()
    print(f'Test executed in {executeTimeF - executeTimeI} seconds')


def process_pair(case, name, pair, iteration, img1, img2, matches_origin, kps1_origin, kps2_origin, std_amount=0.5, thresh=0.05):
    (matches, kps1, kps2) = process_step(case, name, pair, iteration, img1, img2,
                                         matches_origin, kps1_origin, kps2_origin, dist_step, std_amount, thresh)
    (matches, kps1, kps2) = process_step(case, name, pair, iteration,
                                         img1, img2, matches, kps1, kps2, dist_step, 2 * std_amount, thresh, filename_diff='_dist2')
    return process_step(case, name, pair, iteration, img1, img2, matches, kps1, kps2, angle_step, std_amount, thresh, True)


def process_step(case, name, pair, iteration, img1, img2, matches_origin, kps1_origin, kps2_origin, step_fn, std_amount, thresh, use_mean_denominator=False, filename_diff=''):
    print(f'\nremoving outliers with {step_fn.__name__}')
    print(f'initial length of matches: {len(matches_origin)}')

    (kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std) = step_fn(
        matches_origin, kps1_origin, kps2_origin)
    kps_feat_min = np.min(kps_feat_diff)
    kps_feat_max = np.max(kps_feat_diff)
    print(f'min value before removal: {kps_feat_min}')
    print(f'max value before removal: {kps_feat_max}')
    print(f'mean before removal: {kps_feat_diff_mean}')
    print(f'std before removal: {kps_feat_diff_std}')
    write_histogram(
        f'results/matches/{case}_{name}_pair{pair}_iter{iteration}_{step_fn.__name__}_origin{filename_diff}.png', kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, kps_feat_min, kps_feat_max,
        f'{case}, {name}, {step_fn.__name__}, img pair {pair}, iteration {iteration} {filename_diff}', f'Original {step_fn.__name__} ratio', 'Frequency')

    matches, kps1, kps2, removed_matches = remove_fake_matches(
        matches_origin, kps1_origin, kps2_origin, kps_feat_diff,
        kps_feat_diff_mean - (kps_feat_diff_std * std_amount),
        kps_feat_diff_mean + (kps_feat_diff_std * std_amount))
    print(
        f'remaining matches after removal with {std_amount} std: {len(matches)} of {len(matches_origin)} ({len(matches)/len(matches_origin)})')
    write_matches_img(f'results/matches/{case}_{name}_pair{pair}_iter{iteration}_{step_fn.__name__}{filename_diff}.jpg',
                      img1, kps1, img2, kps2, matches[:ARR_LEN])

    (kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std) = step_fn(
        matches, kps1, kps2)
    kps_feat_min = np.min(kps_feat_diff)
    kps_feat_max = np.max(kps_feat_diff)
    print(f'min value after removal: {kps_feat_min}')
    print(f'max value after removal: {kps_feat_max}')
    print(f'mean after removal: {kps_feat_diff_mean}')
    print(f'std after removal: {kps_feat_diff_std}')
    write_histogram(
        f'results/matches/{case}_{name}_pair{pair}_iter{iteration}_{step_fn.__name__}{filename_diff}.png', kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, kps_feat_min, kps_feat_max,
        f'{case}, {name}, {step_fn.__name__}, img pair {pair}, iteration {iteration} {filename_diff}', f'{step_fn.__name__} ratio', 'Frequency')

    is_below_error = stats_eq(
        kps_feat_diff, thresh, kps_feat_diff_mean if use_mean_denominator else None)
    amount_below_error = ft.reduce(op.add, is_below_error, 0)

    print(
        f'matches below {thresh} error: {amount_below_error} of {len(is_below_error)} ({amount_below_error/len(is_below_error)})')

    return matches, kps1, kps2


def stats_eq(stats, thresh=0.05, denominator=None):
    _stats = stats
    if denominator is not None:
        _stats = np.array(
            list(map(ft.partial(op.mul, 1 / denominator), stats)))
    return np.array(list(map(ft.partial(op.ge, 1 + thresh), _stats)))


def write_histogram(path, xs, mean, std, min, max, title=None, xlabel=None, ylabel=None):
    frenquency = len(xs) // 5
    plt.cla()
    plt.hist(xs, frenquency, density=1, color='xkcd:grey')
    # plt.hist(xs, frenquency, range=(mean - (3 * std), mean + (3 * std)), density=1)
    plt.title(title if title is not None else '')
    plt.xlabel(xlabel if xlabel is not None else '')
    plt.ylabel(ylabel if ylabel is not None else '')
    plt.grid(True)

    line_min = plt.axvline(min, c='xkcd:red', ls='-')
    line_std = plt.axvline(mean - std, c='xkcd:purple', ls='--')
    plt.axvline(mean + std, c='xkcd:purple', ls='--')
    line_mean = plt.axvline(mean, c='xkcd:blue', ls='--')
    line_max = plt.axvline(max, c='xkcd:red', ls='-')

    plt.legend((line_min, line_mean, line_std, line_max), ('min = %.4f' %
                                                           min, r'$\mu$ = %.4f' % mean, r'$\sigma$ = %.4f' % std, 'max = %.4f' % max))

    plt.savefig(path)
    # plt.show()


def write_matches_img(path, img1, kps1, img2, kps2, matches):
    img_matches = cv2.drawMatches(
        img1, kps1, img2, kps2, matches,
        outImg=None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
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


def print_matches(matches, kps1, kps2):
    for match in matches:
        q = kps1[match.queryIdx].pt
        t = kps2[match.trainIdx].pt
        print(f'({q[0]}, {q[1]})', f'({t[0]}, {t[1]})')


def dist_step(matches, kps1, kps2):
    center1 = gmt.kps_center(kps1)
    center2 = gmt.kps_center(kps2)

    dist1 = gmt.find_kps_dist(center1, kps1)
    dist2 = gmt.find_kps_dist(center2, kps2)

    diff, diff_mean, diff_std = process_kps_feat(
        matches, dist1, dist2, op.truediv)
    return diff, diff_mean, diff_std


def angle_step(matches, kps1, kps2):
    center1 = gmt.kps_center(kps1)
    center2 = gmt.kps_center(kps2)

    diff, diff_mean, diff_std = process_kps_feat(
        matches, kps1, kps2, gmt.get_kp_angle, center1, center2)
    return diff, diff_mean, diff_std


def kps_fn(matches, kps_feat1, kps_feat2, fn, *fn_args):
    """ Applies fn to all the pairs of matches.

    Passes on fn_args to fn.
    """
    result = np.empty(shape=len(matches))
    i = 0
    for match in matches:
        # result list preserves the ordering from matches list
        q = kps_feat1[match.queryIdx]
        t = kps_feat2[match.trainIdx]
        result[i] = fn(q, t, *fn_args)
        i += 1
    return result


def process_kps_feat(matches, kps_feat1, kps_feat2, fn, *fn_args):
    processed_kps = kps_fn(matches, kps_feat1, kps_feat2, fn, *fn_args)
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
def remove_fake_matches(matches, kps1, kps2, kps_diff, lower_lim, upper_lim):
    j = 0
    new_kps1 = []
    new_kps2 = []
    new_matches = []
    removed_matches = []
    for i in range(len(matches)):
        if kps_diff[i] >= lower_lim and kps_diff[i] <= upper_lim:
            q = kps1[matches[i].queryIdx]
            t = kps2[matches[i].trainIdx]
            new_kps1.append(cv2.KeyPoint(q.pt[0], q.pt[1], q.size,
                                         q.angle, q.response, q.octave, q.class_id))
            new_kps2.append(cv2.KeyPoint(t.pt[0], t.pt[1], t.size,
                                         t.angle, t.response, t.octave, t.class_id))
            new_matches.append(cv2.DMatch(
                j, j, matches[i].imgIdx, matches[i].distance))
            j += 1
        else:
            removed_matches.append(matches[i])
    return np.array(new_matches), np.array(new_kps1), np.array(new_kps2), np.array(removed_matches)


if(__name__ == '__main__'):
    main()
