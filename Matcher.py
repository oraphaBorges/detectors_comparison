import copy as cp
from time import time

import cv2
import numpy as np


class Matcher:
    """ Wrapper class around OpenCV's object detection, description and matching. """

    def __init__(self, alg, img1, img2, norm=None):
        """
        :param alg: feature detector and descriptor from OpenCV Features2D
        :param img1: image 1
        :param img2: image 2
        :param norm: descriptor matcher type (int/enum from cv2)
        """
        self.alg = alg
        self.img1 = img1
        self.img2 = img2
        self.matcher = cv2.DescriptorMatcher.create(
            norm if norm is not None else alg.defaultNorm())
        self.time = None
        self.matches = None
        self.snapshot = None
        self.kps1, self.kps2 = None, None
        self.des1, self.des2 = None, None

    def draw_matches(self, path=None, match_color=(0, 255, 0), point_color=(0, 0, 255)):
        """ Get a side-by-side view of self.img1 and self.img2 and their matches.

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

    def detect_and_compute(self):
        """ Computes keypoints and descriptors for self.img1 and self.img2.

        The time taken to compute everything is stored on self.time.
        """
        self.time = time()
        self.kps1, self.des1 = self.alg.detectAndCompute(self.img1)
        self.kps2, self.des2 = self.alg.detectAndCompute(self.img2)
        self.time = time() - self.time

    def match(self):
        """ Matches computed descriptors. """
        self.matches = self.matcher.match(self.des1, self.des2)

    def backup(self):
        """ Deep-copies all attributes to snapshot (except any previous snapshot). """
        # set self.snapshot to None so the deep copy
        # won't end up nesting all previous snapshots
        # if this is called more than once
        self.snapshot = None
        self.snapshot = cp.deepcopy(self.__dict__)

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
                kps1.append(cp.deepcopy(self.kps1[self.matches[i].queryIdx]))
                des1.append(cp.deepcopy(self.des1[self.matches[i].queryIdx]))

                kps2.append(cp.deepcopy(self.kps2[self.matches[i].trainIdx]))
                des2.append(cp.deepcopy(self.des2[self.matches[i].trainIdx]))

                match = cp.deepcopy(self.matches[i])
                match.queryIdx, match.trainIdx = j, j
                matches.append(match)
                j += 1
            else:
                removed_matches.append(self.matches[i])
        self.matches = matches
        self.kps1, self.kps2 = kps1, kps2
        self.des1, self.des2 = np.array(des1), np.array(des2)
        return removed_matches
