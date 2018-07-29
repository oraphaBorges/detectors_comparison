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

                for name, method in methods.items():
                    print('\n---------------')
                    print(f'Running method: {name}')

                    matcher = Matcher(method, img1, img2)
                    matcher.match()

                    process_by_std(matcher, case, name, pair_number, 1,
                                   dist_step, 1)
                    process_by_std(matcher, case, name, pair_number, 2,
                                   dist_step, 1)

            except IOError as ioerr:
                print(ioerr)
            finally:
                del img1  # img1 will always exist because of glob
                if img2 is not None:
                    del img2
    execute_time_f = time()
    print(f'\nTest executed in {execute_time_f - execute_time_i} seconds')


def process_by_std(matcher, case, name, pair, iteration, step_fn, std_amount,
                   sort=False):
    path_folder = 'results/matches'
    filename = f'{case}_{name}_pair{pair}_iter{iteration}' \
               f'_{step_fn.__name__}_std'
    print(f'\nremoving outliers with {step_fn.__name__}'
          f' over {std_amount} std, iteration {iteration}')

    diff, diff_min, diff_max, diff_mean, diff_std = step_fn(matcher)

    print(f'matches before removal: {len(matcher.matches)}')
    print(f'min value before removal: {diff_min}')
    print(f'max value before removal: {diff_max}')
    print(f'mean before removal: {diff_mean}')
    print(f'std before removal: {diff_std}')

    insert_and_commit((case, name, pair, iteration, step_fn.__name__,
                       diff_min, diff_max, diff_mean, diff_std,
                       len(matcher.matches), None))

    write_boxplot(f'{path_folder}/box_{filename}_original.png',
                  diff, diff_mean, diff_std)

    write_histogram(f'{path_folder}/hist_{filename}_original.png',
                    diff, diff_mean, diff_std, diff_min, diff_max,
                    xlabel=f'{step_fn.__name__}', ylabel='Frequency',
                    xrange=(0, 3) if step_fn == dist_step else (0, 30))

    matcher.backup()
    removed_matches = matcher.filter_by(diff,
                                        diff_mean - (diff_std * std_amount),
                                        diff_mean + (diff_std * std_amount))

    diff, diff_min, diff_max, diff_mean, diff_std = step_fn(matcher)

    print(f'removed {len(removed_matches)} matches'
          f' ({len(removed_matches) / len(matcher.snapshot["matches"])})')
    print(f'matches after removal: {len(matcher.matches)}'
          f' ({len(matcher.matches) / len(matcher.snapshot["matches"])})')
    print(f'min after removal: {diff_min}')
    print(f'max after removal: {diff_max}')
    print(f'mean after removal: {diff_mean}')
    print(f'std after removal: {diff_std}')

    insert_and_commit((case, name, pair, iteration, step_fn.__name__,
                       diff_min, diff_max, diff_mean, diff_std,
                       len(matcher.snapshot['matches']), len(matcher.matches)))

    write_boxplot(f'{path_folder}/box_{filename}_processed.png',
                  diff, diff_mean, diff_std)

    write_histogram(f'{path_folder}/hist_{filename}_processed.png',
                    diff, diff_mean, diff_std, diff_min, diff_max,
                    xlabel=f'{step_fn.__name__}', ylabel='Frequency',
                    xrange=(0, 3) if step_fn == dist_step else (0, 30))

    if sort:
        matcher.sort(diff)

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

    if mean is not None and std is not None and minimum is not None \
            and maximum is not None and xrange is not None:
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


def print_matches(matcher):
    matcher.apply_over_kps(lambda q, t: print(f'({q.pt}), ({t.pt})'))


def dist_step(matcher):
    center1 = gmt.kps_center(matcher.kps1)
    center2 = gmt.kps_center(matcher.kps2)

    dist1 = gmt.find_kps_dist(center1, matcher.kps1)
    dist2 = gmt.find_kps_dist(center2, matcher.kps2)

    diff = matcher.apply_over_matches(dist1, dist2, op.truediv)
    diff_mean = mean(diff)
    diff_std = std(diff, diff_mean)

    return diff, min(diff), max(diff), diff_mean, diff_std


def angle_step(matcher):
    center1 = gmt.kps_center(matcher.kps1)
    center2 = gmt.kps_center(matcher.kps2)

    diff = matcher.apply_over_kps(gmt.get_kp_angle, center1, center2)
    diff_mean = mean(diff)
    diff_std = std(diff, diff_mean)

    return diff, min(diff), max(diff), diff_mean, diff_std


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
