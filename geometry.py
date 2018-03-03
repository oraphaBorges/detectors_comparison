import numpy
from sympy import Point, Line
import cv2

# Finds the image's center
def image_center(image):
    return Point(image.shape[1] / 2, image.shape[0] / 2)


# Finds the angles between the horizontal axis
# and the lines passing through the image center
# and each keypoint
def g_find_kp_angles(image, kps):
    angles = []
    center = image_center(image)
    h_axis = Line(center, center.translate(center.x))
    for kp in kps:
        p = Point(kp.pt[0], kp.pt[1])
        kp_line = Line(center, p)
        angles.append(float(h_axis.angle_between(kp_line)))
    return angles


def angles_dif(angles_img1, angles_img2, matches):
    dif = []
    for match in matches:
        dif.append(angles_img1[match.queryIdx] - angles_img2[match.trainIdx])

    return dif


def remove_fake_matches(matches, dif_angles, mean_angles, angles_std, scales, mean_scale, scale_std):
    new_scales, new_dif_angles = [], []
    for i in range(len(matches)):
        if dif_angles[i] < mean_angles + angles_std and dif_angles[i] > mean_angles - angles_std and scales[i] < mean_scale + scale_std and scales[i] > mean_angles - scale_std:
            new_scales.append(scales[i])
            new_dif_angles.append(dif_angles[i])
    return new_dif_angles, new_scales


# Finds the Key's points Angles
def find_kp_angles(kp1, kp2, matches, center1, center2):
    central_line = Line(center1, center2.translate(2 * center2.x))
    angles = []
    for match in matches:
        p1 = Point(kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1])
        p2 = Point(kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1])
        match_line = Line(p1, p2.translate(2 * center2.x))
        angles.append(float(central_line.angle_between(match_line)))
    return angles


def g_find_scale(image, kps):
    scale = []
    center = image_center(image)
    for kp in kps:
        p = Point(kp.pt[0], kp.pt[1])
        d = center.distance(p)
        scale.append(d)
    return scale


# Finds the ratio of the keypoints scale between images
def find_scale_ratios(img1, kp1, img2, kp2, matches):
    ratios = []
    scale1 = g_find_scale(img1, kp1)
    scale2 = g_find_scale(img2, kp2)
    for match in matches:
        # scale list preserves the ordering from keypoints list
        d1 = scale1[match.queryIdx]
        d2 = scale2[match.trainIdx]
        ratios.append(float(d1 / d2))
    return ratios


# Finds the Scale between images
def find_scale(kp1, kp2, matches, center1, center2):
    scale = []
    for match in matches:
        p1 = Point(kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1])
        p2 = Point(kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1])
        d1 = center1.distance(p1)
        d2 = center2.distance(p2)
        scale.append(float(d1 / d2))
    return scale


def affine_trans(img, angles, scale):
    center = image_center(img)
    m = cv2.getRotationMatrix2D((center.y, center.x), angles, scale)
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
