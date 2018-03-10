import cv2
import numpy as np


def eucl_distance(p1, p2):
    return np.sqrt(np.square(p2[0] - p1[0]) + np.square(p2[1] - p1[1]))


# Returns the vector p1 -> p2
def points_to_vec(p1, p2):
    v = np.empty(shape=2)
    v[0] = p2[0] - p1[0]
    v[1] = p2[1] - p1[1]
    return v

# The mean of the coordinates
# of the keypoints is considered
# the center of the object
def kps_center(kps):
    mean = np.zeros(2)
    for kp in kps:
        mean[0] += kp.pt[0]
        mean[1] += kp.pt[1]
    mean[0] /= len(kps)
    mean[1] /= len(kps)
    return mean

# Finds the angles between the horizontal axis
# and the lines passing through the center of
# the keypoints and each keypoint
def find_kp_angles(image, center, kps):
    angles = np.empty(shape=len(kps))
    center_shifted = np.copy(center)
    center_shifted[0] += center[0]
    h_axis = points_to_vec(center, center_shifted)
    i = 0
    for kp in kps:
        kp_vec = points_to_vec(center, kp.pt)
        # cos(theta) = (u . v) / (|u| * |v|)
        angles[i] = np.dot(h_axis, kp_vec) / \
            (np.linalg.norm(h_axis) * np.linalg.norm(kp_vec))
        angles[i] = np.rad2deg(np.arccos(angles[i]))
        i += 1
    return angles


def angles_diff(angles_img1, angles_img2, matches):
    diff = np.empty(shape=len(matches))
    i = 0
    for match in matches:
        diff[i] = angles_img1[match.queryIdx] - angles_img2[match.trainIdx]
        i += 1
    return diff


def find_kps_dist(center, kps):
    scale = np.empty(shape=len(kps))
    i = 0
    for kp in kps:
        scale[i] = eucl_distance(center, kp.pt)
        i += 1
    return scale


# Finds the ratio of the keypoints scale between images
def kps_ratio(center1, kps1, center2, kps2, matches):
    ratios = np.empty(shape=len(matches))
    dists1 = find_kps_dist(center1, kps1)
    dists2 = find_kps_dist(center2, kps2)
    i = 0
    for match in matches:
        # ratios list preserves the ordering from matches list
        d1 = dists1[match.queryIdx]
        d2 = dists2[match.trainIdx]
        ratios[i] = d1 / d2
        i += 1
    return ratios


# def remove_fake_matches(matches, dif_angles, angles_mean, angles_std, scales, scale_mean, scale_std):
#     new_scales, new_dif_angles = [], []
#     for i in range(len(matches)):
#         if dif_angles[i] < angles_mean + angles_std and dif_angles[i] > angles_mean - angles_std and scales[i] < scale_mean + scale_std and scales[i] > angles_mean - scale_std:
#             new_scales.append(scales[i])
#             new_dif_angles.append(dif_angles[i])
#     return new_dif_angles, new_scales


# def affine_trans(img, angles, scale):
#     center = image_center(img)
#     m = cv2.getRotationMatrix2D((center.y, center.x), angles, scale)
#     return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
