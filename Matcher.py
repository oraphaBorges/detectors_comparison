import copy as cp
from time import time

import cv2
import numpy as np


class Matcher:
    """ Wrapper class around OpenCV's object detection,
    description and matching. """

    def __init__(self, alg, img1, img2, name=None, norm_type=None,
                 cross_check=True):
        """
        :param alg: feature detector and descriptor from OpenCV Features2D
        :param img1: image 1
        :param img2: image 2
        :param name: alg name
        :param norm_type: cv2.DESCRIPTOR_MATCHER_* indicating norm type.
        If None, tries to use default from alg
        :param cross_check: crossCheck parameter of cv2.BFMatcher_create
        """
        self.alg = alg
        self.img1 = img1
        self.img2 = img2
        self.name = name if name is not None else alg.getDefaultName()
        if norm_type is not None:
            self.matcher = cv2.DescriptorMatcher_create(norm_type)
        else:
            self.matcher = cv2.BFMatcher_create(alg.defaultNorm(),
                                                crossCheck=cross_check)
        self.time = None
        self.matches = None
        self.snapshot = None
        self.kps1, self.kps2 = None, None
        self.des1, self.des2 = None, None

    def detect_and_compute(self):
        """ Computes keypoints and descriptors for self.img1 and self.img2.

        The time taken to compute everything is stored on self.time.
        """
        self.time = time()
        self.kps1, self.des1 = self.alg.detectAndCompute(self.img1, mask=None)
        self.kps2, self.des2 = self.alg.detectAndCompute(self.img2, mask=None)
        self.time = time() - self.time

    def match(self):
        """ Matches computed descriptors.

        Calls self.detect_and_compute() if any of self
        kps1, des1, kps2, des2 is None (e.g. if you haven't).
        """
        if self.kps1 is None or self.des1 is None \
                or self.kps2 is None or self.des2 is None:
            self.detect_and_compute()
        self.matches = self.matcher.match(self.des1, self.des2)

    def sort(self, base_array=None):
        """ Sorts self.matches according to their distance attribute
        or base array parameter.

        :param base_array: array-like object with the same length
        as self.matches with scalar values to compute an ordering
        :return: the indexes that have been used to sort the matches
        """
        if base_array is None:
            base_array = [match.distance for match in self.matches]
        indxs = np.argsort(base_array)
        self.matches = self.matches[indxs]
        return indxs

    def draw_matches(self, path=None,
                     match_color=(0, 255, 0), point_color=(0, 0, 255)):
        """ Get a side-by-side view of
        self.img1 and self.img2 and their matches.

        :param path: optional path to save the image in the filesystem
        :param match_color: BGR color of matches
        :param point_color: BGR color of single keypoints
        :return: image with self.img1 and self.img2 and their matches drawn
        """
        img_with_matches = cv2.drawMatches(
            self.img1, self.kps1, self.img2, self.kps2, self.matches,
            outImg=None, matchColor=match_color, singlePointColor=point_color)
        if path is not None:
            cv2.imwrite(path, img_with_matches)
        return img_with_matches

    def backup(self):
        """ Copies all attributes to self.snapshot
        (except any previous snapshot). """
        # set self.snapshot to None so the copy
        # won't end up nesting all previous snapshots
        # if this is called more than once
        self.snapshot = None
        self.snapshot = cp.copy(self.__dict__)

    def apply_over_kps(self, fn, *fn_args):
        """ Utility method that calls self.apply_over_matches with
        self.kps1 and self.kps2 as arguments.

        See self.apply_over_matches for more details.
        """
        return self.apply_over_matches(self.kps1, self.kps2, fn, *fn_args)

    def apply_over_matches(self, meas1, meas2, fn, *fn_args):
        """ Calls fn for all pairs of meas1 and meas2 indexed by self.matches.

        Passes on fn_args to fn.
        Note that the result depends on the current ordering of matches.

        :param fn: function what will receive each pair
        :param fn_args: optional additional arguments to fn
        :return: all returned values from fn in a list
        """
        measures = []
        for match in self.matches:
            q = meas1[match.queryIdx]
            t = meas2[match.trainIdx]
            measures.append(fn(q, t, *fn_args))
        return measures

    def filter_by(self, measure, lower_bound, upper_bound):
        """ Filters and *overwrites* self keypoints, descriptors and matches
        based on the passed measure and lower and upper bounds.

        :param measure: some computed value over the image keypoints
        :param lower_bound: measure desired lower bound
        :param upper_bound: measure desired upper bound
        :return: the matches removed (old keypoints and descriptors can
        be queried on self.snapshot if a self.backup() was previously called)
        """
        j = 0
        matches = []
        kps1, des1 = [], []
        kps2, des2 = [], []
        removed_matches = []
        for i in range(len(self.matches)):
            if lower_bound <= measure[i] <= upper_bound:
                # why were we copying?
                kps1.append(self.kps1[self.matches[i].queryIdx])
                des1.append(self.des1[self.matches[i].queryIdx])

                kps2.append(self.kps2[self.matches[i].trainIdx])
                des2.append(self.des2[self.matches[i].trainIdx])

                match = self.matches[i]  # ye I know, keystrokes
                match.queryIdx, match.trainIdx = j, j
                matches.append(match)
                j += 1
            else:
                removed_matches.append(self.matches[i])
        self.matches = matches
        self.kps1, self.kps2 = kps1, kps2
        self.des1, self.des2 = np.array(des1), np.array(des2)
        return removed_matches
