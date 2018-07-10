import operator as op
import sqlite3
from glob import glob
from time import time, strftime

import cv2
import numpy as np
from matplotlib import pyplot as plt

import geometry as gmt

ARR_LEN = 10
TABLE_NAME = f'stats_{strftime("%y%m%d_%H%M%S")}'


def main():
    execute_time_i = time()

    orb = cv2.ORB.create(nfeatures=5000)
    akaze = cv2.AKAZE.create()
    brisk = cv2.BRISK.create()
    # sift = cv2.xfeatures2d.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()

    methods = {
        'ORB': orb,
        'AKAZE': akaze,
        'BRISK': brisk,
        # 'SIFT': sift,
        # 'SURF': surf
    }

    cases = [
        # 'Different Object',
        'Same Object, Same Scale',
        # 'Same Object, Different Scale'
    ]

    for case in cases:
        print('\n\n\n####################')
        print(f'Running case: {case}')

        path_images = glob(f'photos/{case}/*a.jpg').sort()
        qtty_images = len(path_images)
        curr_image = 0
        for path_a in path_images:
            path_prefix = path_a.replace('a.jpg', '')
            pair_number = path_prefix.split('/')[-1]
            print('\n\n+++++++++++++++')
            print(f'Pair {pair_number} ({curr_image}/{qtty_images - 1})')
            curr_image += 1

            try:
                img1 = cv2.imread(f'{path_prefix}a.jpg', 0)
                img2 = cv2.imread(f'{path_prefix}b.jpg', 0)

                for name, method in methods.items():
                    print('\n---------------')
                    print(f'Running method: {name}')

                    matches_origin, kps1_origin, kps2_origin, _ = get_stats(method, img1, img2)

                    # write_matches_img(f'results/matches/{case}_{name}_pair{pair}_{step_fn.__name__}'
                    #                   f'_iter{iteration}_original_matches.png',
                    #                   img1, kps1_origin, img2, kps2_origin, matches_origin[:ARR_LEN])

                    # matches, kps1, kps2 = process_step(case, name, pair, 1,
                    #                                    matches_origin, kps1_origin, kps2_origin, dist_step, 1)
                    # process_step(case, name, pair, 2, matches, kps1, kps2, dist_step, 1)

                    matches, kps1, kps2 = process_by_med(case, name, pair_number, 1,
                                                         matches_origin, kps1_origin, kps2_origin, dist_step)
                    matches, kps1, kps2 = process_by_med(case, name, pair_number, 2, matches, kps1, kps2, dist_step)
                    process_by_std(case, name, pair_number, 3, matches, kps1, kps2, dist_step, 1)

            except IOError as ioerr:
                print(ioerr)
            finally:
                del img1  # img1 will always exist because of glob
                if img2 is not None:
                    del img2
    execute_time_f = time()
    print(f'\nTest executed in {execute_time_f - execute_time_i} seconds')


def process_by_med(case, name, pair, iteration, matches_origin, kps1_origin, kps2_origin, step_fn):
    path_folder = 'results/matches'
    filename = f'{case}_{name}_pair{pair}_iter{iteration}_{step_fn.__name__}_med'
    print(f'\nremoving outliers with {step_fn.__name__} over boxes, iteration {iteration}')

    kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std = step_fn(matches_origin, kps1_origin, kps2_origin)
    kps_feat_min = np.min(kps_feat_diff)
    kps_feat_max = np.max(kps_feat_diff)
    kps_feat_qs = np.percentile(kps_feat_diff, [25, 50, 70])
    print(f'matches before removal: {len(matches_origin)}')
    print(f'min value before removal: {kps_feat_min}')
    print(f'max value before removal: {kps_feat_max}')
    print(f'mean before removal: {kps_feat_diff_mean}')
    print(f'std before removal: {kps_feat_diff_std}')
    print(f'medians before removal: {kps_feat_qs}')

    insert_and_commit((case, name, pair, iteration, step_fn.__name__, kps_feat_min, kps_feat_max, kps_feat_diff_mean,
                       kps_feat_diff_std, len(matches_origin), None, None, None, None, None, None, None, None))

    write_boxplot(f'{path_folder}/box_{filename}_original.png',
                  kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, [kps_feat_qs[1]])

    write_histogram(f'{path_folder}/hist_{filename}_original.png',
                    kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, kps_feat_min, kps_feat_max,
                    xlabel=f'{step_fn.__name__}', ylabel='Frequency',
                    xrange=(0, 3) if step_fn == dist_step else (0, 30))

    matches, kps1, kps2, removed_matches = remove_fake_matches(
        matches_origin, kps1_origin, kps2_origin, kps_feat_diff,
        kps_feat_qs[0], kps_feat_qs[2])

    kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std = step_fn(matches, kps1, kps2)
    kps_feat_min = np.min(kps_feat_diff)
    kps_feat_max = np.max(kps_feat_diff)
    kps_feat_qs = np.percentile(kps_feat_diff, [25, 50, 70])
    print(f'matches after removal: {len(matches)}')
    print(f'min value after removal: {kps_feat_min}')
    print(f'max value after removal: {kps_feat_max}')
    print(f'mean after removal: {kps_feat_diff_mean}')
    print(f'std after removal: {kps_feat_diff_std}')
    print(f'medians after removal: {kps_feat_qs}')

    insert_and_commit((case, name, pair, iteration, step_fn.__name__, kps_feat_min, kps_feat_max, kps_feat_diff_mean,
                       kps_feat_diff_std, len(matches_origin), len(matches), None, None, None, None, None, None, None))

    write_boxplot(f'{path_folder}/box_{filename}_processed.png',
                  kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, [kps_feat_qs[1]])

    write_histogram(f'{path_folder}/hist_{filename}_processed.png',
                    kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, kps_feat_min, kps_feat_max,
                    xlabel=f'{step_fn.__name__}', ylabel='Frequency',
                    xrange=(0, 3) if step_fn == dist_step else (0, 30))

    return matches, kps1, kps2


def process_by_std(case, name, pair, iteration, matches_origin, kps1_origin, kps2_origin, step_fn, std_amount,
                   use_mean_denominator=False):
    path_folder = 'results/matches'
    filename = f'{case}_{name}_pair{pair}_iter{iteration}_{step_fn.__name__}_std'
    print(f'\nremoving outliers with {step_fn.__name__} over {std_amount} std, iteration {iteration}')

    kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std = step_fn(matches_origin, kps1_origin, kps2_origin)
    kps_feat_min = np.min(kps_feat_diff)
    kps_feat_max = np.max(kps_feat_diff)

    # Print only on the first time that we call this method
    # if iteration == 1:
    amount_below_5 = amount_stats_within(kps_feat_diff, 0.05, kps_feat_diff_mean if use_mean_denominator else None)
    amount_below_10 = amount_stats_within(kps_feat_diff, 0.1, kps_feat_diff_mean if use_mean_denominator else None)
    amount_below_20 = amount_stats_within(kps_feat_diff, 0.2, kps_feat_diff_mean if use_mean_denominator else None)
    ratio5 = amount_below_5 / len(kps_feat_diff)
    ratio10 = amount_below_10 / len(kps_feat_diff)
    ratio20 = amount_below_20 / len(kps_feat_diff)
    kps_feat_diff_beyond_std = \
        len(kps_feat_diff) - amount_within_std(kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std)

    print(f'matches before removal: {len(matches_origin)}')
    print(f'min value before removal: {kps_feat_min}')
    print(f'max value before removal: {kps_feat_max}')
    print(f'mean before removal: {kps_feat_diff_mean}')
    print(f'std before removal: {kps_feat_diff_std}')
    print(f'amount beyond std before removal: {kps_feat_diff_beyond_std}')

    print(f'matches below 0.05 error before removal: {amount_below_5} of {len(kps_feat_diff)} ({ratio5})')
    print(f'matches below 0.10 error before removal: {amount_below_10} of {len(kps_feat_diff)} ({ratio10})')
    print(f'matches below 0.20 error before removal: {amount_below_20} of {len(kps_feat_diff)} ({ratio20})')

    insert_and_commit((case, name, pair, iteration, step_fn.__name__, kps_feat_min, kps_feat_max, kps_feat_diff_mean,
                       kps_feat_diff_std, len(matches_origin), None, int(kps_feat_diff_beyond_std),
                       float(amount_below_5), float(amount_below_10), float(amount_below_20), ratio5, ratio10, ratio20))

    write_boxplot(f'{path_folder}/box_{filename}_original.png',
                  kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std)

    write_histogram(f'{path_folder}/hist_{filename}_original.png',
                    kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, kps_feat_min, kps_feat_max,
                    xlabel=f'{step_fn.__name__}', ylabel='Frequency',
                    xrange=(0, 3) if step_fn == dist_step else (0, 30))

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
    ratio10 = amount_below_10 / len(kps_feat_diff)
    ratio20 = amount_below_20 / len(kps_feat_diff)
    kps_feat_diff_beyond_std = \
        len(kps_feat_diff) - amount_within_std(kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std)

    print(f'removed {len(removed_matches)} matches ({len(removed_matches) / len(matches_origin)})')
    print(f'matches after removal: {len(matches)} ({len(matches) / len(matches_origin)})')
    print(f'min value after removal: {kps_feat_min}')
    print(f'max value after removal: {kps_feat_max}')
    print(f'mean after removal: {kps_feat_diff_mean}')
    print(f'std after removal: {kps_feat_diff_std}')
    print(f'amount beyond std after removal: {kps_feat_diff_beyond_std}')

    print(f'matches below 0.05 error after removal: {amount_below_5} of {len(kps_feat_diff)} ({ratio5})')
    print(f'matches below 0.10 error after removal: {amount_below_10} of {len(kps_feat_diff)} ({ratio10})')
    print(f'matches below 0.20 error after removal: {amount_below_20} of {len(kps_feat_diff)} ({ratio20})')

    insert_and_commit((case, name, pair, iteration, step_fn.__name__, kps_feat_min, kps_feat_max, kps_feat_diff_mean,
                       kps_feat_diff_std, len(matches_origin), len(matches), int(kps_feat_diff_beyond_std),
                       float(amount_below_5), float(amount_below_10), float(amount_below_20), ratio5, ratio10, ratio20))

    write_boxplot(f'{path_folder}/box_{filename}_processed.png',
                  kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std)

    write_histogram(f'{path_folder}/hist_{filename}_processed.png',
        kps_feat_diff, kps_feat_diff_mean, kps_feat_diff_std, kps_feat_min, kps_feat_max,
        xlabel=f'{step_fn.__name__}', ylabel='Frequency', xrange=(0, 3) if step_fn == dist_step else (0, 30))

    return matches, kps1, kps2


def amount_stats_within(stats, thresh=0.05, denominator=None):
    is_below = stats_within(stats, thresh, denominator)
    return np.sum(is_below)


def stats_within(stats, thresh=0.05, denominator=None):
    """ Tests if every stat in stats is within given threshold.

    For ratios, tests if every element is <= thresh.
    Else, like for angles that are a difference, tests if
    every element over the mean is <= thresh.
    """
    _stats = stats
    if denominator is not None:
        _stats = stats / denominator
    return _stats < (1 + thresh)


def amount_within_std(stats, mean, std):
    return np.sum(np.logical_and(stats >= mean - std, stats <= mean + std))


def write_boxplot(path, xs, mean=None, std=None, usermedians=None, title=None):
    plt.cla()
    plt.boxplot(xs, usermedians=usermedians)
    plt.title(title if title is not None else '')
    plt.grid(True)

    if mean is not None and std is not None:
        line_std = plt.axhline(mean - std, c='xkcd:purple', ls='--')
        line_std.set_label(r'$\sigma$ = %.4f' % std)

        plt.axhline(mean + std, c='xkcd:purple', ls='--')

        line_mean = plt.axhline(mean, c='xkcd:blue', ls='--')
        line_mean.set_label(r'$\mu$ = %.4f' % mean)

        plt.legend()

    plt.savefig(path)


def write_histogram(path, xs,
                    mean=None, std=None, minimum=None, maximum=None, xrange=None, title=None, xlabel=None, ylabel=None):
    plt.cla()
    plt.hist(xs, bins=25, range=xrange, density=True, color='xkcd:grey')
    plt.title(title if title is not None else '')
    plt.xlabel(xlabel if xlabel is not None else '')
    plt.ylabel(ylabel if ylabel is not None else '')
    plt.grid(True)

    if mean is not None and std is not None and minimum is not None and maximum is not None and xrange is not None:
        if minimum >= xrange[0]:
            line_min = plt.axvline(minimum, c='xkcd:red', ls='-')
            line_min.set_label('min = %.4f' % minimum)

        if xrange[0] <= mean - std <= xrange[1]:
            line_std = plt.axvline(mean - std, c='xkcd:purple', ls='--')
            line_std.set_label(r'$\sigma$ = %.4f' % std)

        if xrange[0] <= mean + std <= xrange[1]:
            plt.axvline(mean + std, c='xkcd:purple', ls='--')

        if xrange[0] <= mean <= xrange[1]:
            line_mean = plt.axvline(mean, c='xkcd:blue', ls='--')
            line_mean.set_label(r'$\mu$ = %.4f' % mean)

        if maximum <= xrange[1]:
            line_max = plt.axvline(maximum, c='xkcd:red', ls='-')
            line_max.set_label('max = %.4f' % maximum)

    plt.legend()

    plt.savefig(path)


def write_matches_img(path, img1, kps1, img2, kps2, matches):
    img_matches = cv2.drawMatches(
        img1, kps1, img2, kps2, matches,
        outImg=None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
    cv2.imwrite(path, img_matches)


def get_stats(method, img1, img2):
    time_i = time()
    kps1, des1 = method.detectAndCompute(img1, None)
    kps2, des2 = method.detectAndCompute(img2, None)

    # create BFMatcher object and
    # match descriptors (query, train)
    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(des1, des2)
    time_f = time()

    return np.array(matches), np.array(kps1), np.array(kps2), time_f - time_i


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
            matches_origin, matches, amount_beyond_std, amount_below_5, amount_below_10, amount_below_20,
            ratio5, ratio10, ratio20
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                amount_beyond_std integer,
                amount_below_5 integer,
                amount_below_10 integer,
                amount_below_20 integer,
                ratio5 real,
                ratio10 real,
                ratio20 real
            );
            """
        )
        main()
