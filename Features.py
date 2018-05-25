import cv2
import sqlite3
import numpy as np
import operator as op
import functools as ft
import geometry as gmt
from time import time, strftime
from matplotlib import pyplot as plt

ARR_LEN = 10
NUM_OF_PAIRS = 1
TABLE_NAME = f'stats_{strftime("%y%m%d_%H%M%S")}'


def main():
    execute_time_i = time()

    # ORB = cv2.ORB.create(nfeatures=5000)
    AKAZE = cv2.AKAZE.create()
    BRISK = cv2.BRISK.create()
    SIFT = cv2.xfeatures2d.SIFT_create()
    SURF = cv2.xfeatures2d.SURF_create()

    methods = {
        # 'ORB': ORB,
        'AKAZE': AKAZE,
        # 'BRISK': BRISK,
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

                matches_origin, kps1_origin, kps2_origin, _ = get_stats(method, img1, img2)

                # run dist step twice without overwriting original stats
                matches, kps1, kps2 = process_step(case, name, pair, 1, img1, img2,
                                                   matches_origin, kps1_origin, kps2_origin, dist_step, 1 / 2)
                process_step(case, name, pair, 2, img1, img2, matches, kps1, kps2, dist_step, 1)

                # run angle step twice without overwriting original stats
                matches, kps1, kps2 = process_step(case, name, pair, 1, img1, img2,
                                                   matches_origin, kps1_origin, kps2_origin, angle_step, 1 / 2,
                                                   use_mean_denominator=True)
                process_step(case, name, pair, 2, img1, img2, matches, kps1, kps2, angle_step, 1,
                             use_mean_denominator=True)

            del img1
            del img2
    execute_time_f = time()
    print(f'\nTest executed in {execute_time_f - execute_time_i} seconds')


def process_step(case, name, pair, iteration, img1, img2, matches_origin, kps1_origin, kps2_origin, step_fn, std_amount,
                 use_mean_denominator=False):
    print(f'\nremoving outliers with {step_fn.__name__}, iteration {iteration}')

    kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std = step_fn(matches_origin, kps1_origin, kps2_origin)
    kps_feat_min = np.min(kps_feat_diff)
    kps_feat_max = np.max(kps_feat_diff)

    # Print only on the first time that we call this method
    if iteration == 1:
        amount_below_5 = amount_stats_within(kps_feat_diff, 0.05, kps_feat_diff_mean if use_mean_denominator else None)
        amount_below_10 = amount_stats_within(kps_feat_diff, 0.1, kps_feat_diff_mean if use_mean_denominator else None)
        amount_below_20 = amount_stats_within(kps_feat_diff, 0.2, kps_feat_diff_mean if use_mean_denominator else None)
        # len(kps_feat_diff) == len(matches)
        ratio5 = amount_below_5 / len(kps_feat_diff)

        print(f'min value before removal: {kps_feat_min}')
        print(f'max value before removal: {kps_feat_max}')
        print(f'mean before removal: {kps_feat_diff_mean}')
        print(f'std before removal: {kps_feat_diff_std}')

        print(f'matches below 0.05 error before removal: {amount_below_5} of {len(kps_feat_diff)} ({ratio5})')
        print(f'matches below 0.10 error before removal: {amount_below_10} of {len(kps_feat_diff)}'
              f' ({amount_below_10/len(kps_feat_diff)})')
        print(f'matches below 0.20 error before removal: {amount_below_20} of {len(kps_feat_diff)}'
              f' ({amount_below_20/len(kps_feat_diff)})')

        # `ratio >= 0.5` means we believe its the same object
        # iteration 0 means we didn't remove anything yet
        insert_and_commit((case, name, pair, 0, step_fn.__name__, kps_feat_min, kps_feat_max, kps_feat_diff_mean,
                           kps_feat_diff_std, len(matches_origin), None,
                           float(amount_below_5), float(amount_below_10), float(amount_below_20), int(ratio5 >= 0.5)))

    write_histogram(
        f'results/matches/{case}_{name}_pair{pair}_{step_fn.__name__}_iter{iteration}_original.png',
        kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, kps_feat_min, kps_feat_max,
        title=f'{case}, {name}, {step_fn.__name__}, pair {pair}, iter {iteration}, original',
        xlabel=f'{step_fn.__name__}', ylabel='Frequency')

    write_matches_img(f'results/matches/{case}_{name}_pair{pair}_{step_fn.__name__}'
                      f'_iter{iteration}_original_matches.png',
                      img1, kps1_origin, img2, kps2_origin, matches_origin[:ARR_LEN])

    matches, kps1, kps2, removed_matches = remove_fake_matches(
        matches_origin, kps1_origin, kps2_origin, kps_feat_diff,
        kps_feat_diff_mean - (kps_feat_diff_std * std_amount),
        kps_feat_diff_mean + (kps_feat_diff_std * std_amount))

    kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std = step_fn(matches, kps1, kps2)
    kps_feat_min = np.min(kps_feat_diff)
    kps_feat_max = np.max(kps_feat_diff)

    amount_below_5 = amount_stats_within(kps_feat_diff, 0.05, kps_feat_diff_mean if use_mean_denominator else None)
    amount_below_10 = amount_stats_within(kps_feat_diff, 0.1, kps_feat_diff_mean if use_mean_denominator else None)
    amount_below_20 = amount_stats_within(kps_feat_diff, 0.2, kps_feat_diff_mean if use_mean_denominator else None)
    ratio5 = amount_below_5 / len(kps_feat_diff)

    print(f'min value after removal: {kps_feat_min}')
    print(f'max value after removal: {kps_feat_max}')
    print(f'mean after removal: {kps_feat_diff_mean}')
    print(f'std after removal: {kps_feat_diff_std}')
    print(f'remaining matches after removal with {std_amount} std: {len(matches)} of {len(matches_origin)}'
          f' ({len(matches)/len(matches_origin)})')

    print(f'matches below 0.05 error after removal: {amount_below_5} of {len(kps_feat_diff)} ({ratio5})')
    print(f'matches below 0.10 error after removal: {amount_below_10} of {len(kps_feat_diff)}'
          f' ({amount_below_10/len(kps_feat_diff)})')
    print(f'matches below 0.20 error after removal: {amount_below_20} of {len(kps_feat_diff)}'
          f' ({amount_below_20/len(kps_feat_diff)})')

    insert_and_commit((case, name, pair, iteration, step_fn.__name__, kps_feat_min, kps_feat_max, kps_feat_diff_mean,
                       kps_feat_diff_std, len(matches_origin), len(matches),
                       float(amount_below_5), float(amount_below_10), float(amount_below_20), int(ratio5 >= 0.5)))

    write_histogram(
        f'results/matches/{case}_{name}_pair{pair}_{step_fn.__name__}_iter{iteration}_processed.png',
        kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, kps_feat_min, kps_feat_max,
        title=f'{case}, {name}, {step_fn.__name__}, pair {pair}, iter {iteration}, processed',
        xlabel=f'{step_fn.__name__}', ylabel='Frequency')

    return matches, kps1, kps2


def amount_stats_within(stats, thresh=0.05, denominator=None):
    is_below = stats_within(stats, thresh, denominator)
    return sum(is_below)


def stats_within(stats, thresh=0.05, denominator=None):
    """ Tests if every stat in stats is within given threshold.

    For ratios, tests if every element is <= thresh.
    Else, like for angles that are a difference, tests if
    every element over the mean is <= thresh.
    """
    _stats = stats
    if denominator is not None:
        _stats = np.array(
            list(map(ft.partial(op.mul, 1 / denominator), stats)))
    return np.array(list(map(ft.partial(op.ge, 1 + thresh), _stats)))


def write_histogram(path, xs, mean, std, min, max, title=None, xlabel=None, ylabel=None):
    frequency = len(xs) // 5
    plt.cla()
    plt.hist(xs, frequency, density=1, color='xkcd:grey')
    # plt.hist(xs, frequency, range=(mean - (3 * std), mean + (3 * std)), density=1)
    plt.title(title if title is not None else '')
    plt.xlabel(xlabel if xlabel is not None else '')
    plt.ylabel(ylabel if ylabel is not None else '')
    plt.grid(True)

    line_min = plt.axvline(min, c='xkcd:red', ls='-')
    line_std = plt.axvline(mean - std, c='xkcd:purple', ls='--')
    plt.axvline(mean + std, c='xkcd:purple', ls='--')
    line_mean = plt.axvline(mean, c='xkcd:blue', ls='--')
    line_max = plt.axvline(max, c='xkcd:red', ls='-')

    plt.legend((line_min, line_mean, line_std, line_max),
               ('min = %.4f' % min, r'$\mu$ = %.4f' % mean, r'$\sigma$ = %.4f' % std, 'max = %.4f' % max))

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

    return np.array(matches), np.array(kps1), np.array(kps2), timeF - timeI


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
        if lower_lim <= kps_diff[i] <= upper_lim:
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


def insert_and_commit(values):
    cursor.execute(
        """INSERT INTO {} (
            kase, name, pair, iteration, status,
            kps_feat_min, kps_feat_max, kps_feat_diff_mean, kps_feat_diff_std,
            matches_origin, matches, amount_below_5, amount_below_10, amount_below_20, considered_eq
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """.format(TABLE_NAME), values)
    conn.commit()


if __name__ == '__main__':
    with sqlite3.connect("db.sqlite3") as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            create table {TABLE_NAME} (
                kase text,
                name text,
                pair integer,
                iteration integer,
                status text,
                kps_feat_min real,
                kps_feat_max real,
                kps_feat_diff_mean real,
                kps_feat_diff_std real,
                matches_origin integer,
                matches integer,
                amount_below_5 integer,
                amount_below_10 integer,
                amount_below_20 integer,
                considered_eq integer
            );
            """
        )
        main()
