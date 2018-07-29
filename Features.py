import operator as op
import sqlite3
from glob import glob
from statistics import mean, pstdev as std
from time import time, strftime

import cv2
import numpy as np
from matplotlib import pyplot as plt

import geometry as gmt
from Matcher import Matcher

ECHO_LEN = 400
TABLE_NAME = f'stats_{strftime("%y%m%d_%H%M%S")}'


def main():
    execute_time_i = time()

    orb = cv2.ORB.create(nfeatures=5000)
    akaze = cv2.AKAZE.create()
    brisk = cv2.BRISK.create()
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()

    methods = {
        'ORB': orb,
        # 'AKAZE': akaze,
        # 'BRISK': brisk,
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

        path_images = glob(f'photos/{case}/*a.jpg')
        path_images.sort()
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

                for method_name, method in methods.items():
                    print('\n---------------')
                    print(f'Running method: {method_name}')

                    matcher = Matcher(method, img1, img2, name=method_name)
                    matcher.match()

                    process_by_fn(matcher, case, pair_number, 1, dist_std)
                    process_by_fn(matcher, case, pair_number, 2, dist_std)

            except IOError as ioerr:
                print(ioerr)
            finally:
                del img1  # img1 will always exist because of glob
                if img2 is not None:
                    del img2
    execute_time_f = time()
    print(f'\nTest executed in {execute_time_f - execute_time_i} seconds')


def process_by_fn(matcher, case_id, pair_id, exec_id, fn, *fn_args,
                  sort=False, folder_path='results/matches'):
    """

    fn must return a 7-elements-long tuple containing:
    - the arguments of Matcher.filter_by in the default order
    (measure, lower bound, upper bound);
    - the min, max, mean and standard deviation of measure.

    :param matcher: a Matcher object already matched
    :param case_id: an ID for the case being treated
    :param pair_id: an ID for the pair of images being treated
    :param exec_id: an ID for the execution being made
    :param fn: a callable that returns the arguments for Matcher.filter_by()
    :param fn_args: optional additional arguments to fn
    :param sort: if should sort matches after this execution
    :param folder_path: optional folder path to save results
    :return: removed matches
    """
    print(f'\nremoving outliers with {fn.__name__} and args {fn_args}')
    filename = f'{case_id}_{matcher.name}_pair{pair_id}_exec{exec_id}' \
               f'_{fn.__name__}'

    measure, lo_bound, up_bound, meas_min, meas_max, meas_mean, meas_std = fn(
        matcher, *fn_args)

    print(f'matches before removal: {len(matcher.matches)}')
    print(f'min before removal: {meas_min}')
    print(f'max before removal: {meas_max}')
    print(f'mean before removal: {meas_mean}')
    print(f'std before removal: {meas_std}')

    insert_and_commit((case_id, matcher.name, pair_id, exec_id, fn.__name__,
                       meas_min, meas_max, meas_mean, meas_std,
                       len(matcher.matches), None))

    write_boxplot(f'{folder_path}/box_{filename}_original.png',
                  measure, meas_mean, meas_std)

    write_histogram(f'{folder_path}/hist_{filename}_original.png',
                    measure, meas_mean, meas_std, meas_min, meas_max,
                    xlabel=f'{fn.__name__}', ylabel='Frequency')

    matcher.backup()
    removed_matches = matcher.filter_by(measure, lo_bound, up_bound)

    measure, lo_bound, up_bound, meas_min, meas_max, meas_mean, meas_std = fn(
        matcher, *fn_args)

    print(f'removed {len(removed_matches)} matches'
          f' ({len(removed_matches) / len(matcher.snapshot["matches"])})')
    print(f'matches after removal: {len(matcher.matches)}'
          f' ({len(matcher.matches) / len(matcher.snapshot["matches"])})')
    print(f'min after removal: {meas_min}')
    print(f'max after removal: {meas_max}')
    print(f'mean after removal: {meas_mean}')
    print(f'std after removal: {meas_std}')

    insert_and_commit((case_id, matcher.name, pair_id, exec_id, fn.__name__,
                       meas_min, meas_max, meas_mean, meas_std,
                       len(matcher.snapshot['matches']), len(matcher.matches)))

    write_boxplot(f'{folder_path}/box_{filename}_processed.png',
                  measure, meas_mean, meas_std)

    write_histogram(f'{folder_path}/hist_{filename}_processed.png',
                    measure, meas_mean, meas_std, meas_min, meas_max,
                    xlabel=f'{fn.__name__}', ylabel='Frequency')

    if sort:
        matcher.sort(measure)

    return removed_matches


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
                    mean=None, std=None, minimum=None, maximum=None,
                    xrange=None, title=None, xlabel=None, ylabel=None):
    plt.cla()
    plt.hist(xs, bins=25, range=xrange, density=True, color='xkcd:grey')
    plt.title(title if title is not None else '')
    plt.xlabel(xlabel if xlabel is not None else '')
    plt.ylabel(ylabel if ylabel is not None else '')
    plt.grid(True)

    if xrange is None:
        xrange = (min(xs), max(xs))

    if mean is not None and std is not None:
        if xrange[0] <= mean - std <= xrange[1]:
            line_std = plt.axvline(mean - std, c='xkcd:purple', ls='--')
            line_std.set_label(r'$\sigma$ = %.4f' % std)

        if xrange[0] <= mean + std <= xrange[1]:
            plt.axvline(mean + std, c='xkcd:purple', ls='--')

        if xrange[0] <= mean <= xrange[1]:
            line_mean = plt.axvline(mean, c='xkcd:blue', ls='--')
            line_mean.set_label(r'$\mu$ = %.4f' % mean)

    if minimum is not None and maximum is not None:
        if minimum >= xrange[0]:
            line_min = plt.axvline(minimum, c='xkcd:red', ls='-')
            line_min.set_label('min = %.4f' % minimum)

        if maximum <= xrange[1]:
            line_max = plt.axvline(maximum, c='xkcd:red', ls='-')
            line_max.set_label('max = %.4f' % maximum)

    plt.legend()

    plt.savefig(path)


def print_matches(matcher):
    matcher.apply_over_kps(lambda q, t: print(f'({q.pt}), ({t.pt})'))


def dist_std(matcher, std_multiplier=1):
    diff = [match.distance for match in matcher.matches]
    diff_min = min(diff)
    diff_max = max(diff)
    diff_mean = mean(diff)
    diff_std = std(diff, diff_mean)

    lo_bound = diff_mean - (diff_std * std_multiplier)
    hi_bound = diff_mean + (diff_std * std_multiplier)

    return diff, lo_bound, hi_bound, diff_min, diff_max, diff_mean, diff_std


def dist_center_std(matcher, std_multiplier=1):
    center1 = gmt.kps_center(matcher.kps1)
    center2 = gmt.kps_center(matcher.kps2)

    dist1 = gmt.find_kps_dist(center1, matcher.kps1)
    dist2 = gmt.find_kps_dist(center2, matcher.kps2)

    diff = matcher.apply_over_matches(dist1, dist2, op.truediv)
    diff_min = min(diff)
    diff_max = max(diff)
    diff_mean = mean(diff)
    diff_std = std(diff, diff_mean)

    lo_bound = diff_mean - (diff_std * std_multiplier)
    hi_bound = diff_mean + (diff_std * std_multiplier)

    return diff, lo_bound, hi_bound, diff_min, diff_max, diff_mean, diff_std


def angle_std(matcher, std_multiplier=1):
    center1 = gmt.kps_center(matcher.kps1)
    center2 = gmt.kps_center(matcher.kps2)

    diff = matcher.apply_over_kps(gmt.get_kp_angle, center1, center2)
    diff_min = min(diff)
    diff_max = max(diff)
    diff_mean = mean(diff)
    diff_std = std(diff, diff_mean)

    lo_bound = diff_mean - (diff_std * std_multiplier)
    hi_bound = diff_mean + (diff_std * std_multiplier)

    return diff, lo_bound, hi_bound, diff_min, diff_max, diff_mean, diff_std


def insert_and_commit(values):
    cursor.execute(
        """INSERT INTO {} (
            kase, name, pair, iteration, status,
            kps_feat_min, kps_feat_max, kps_feat_diff_mean, kps_feat_diff_std,
            matches_origin, matches
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                matches integer
            );
            """
        )
        main()
