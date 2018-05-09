import cv2
import numpy as np


def eucl_distance(p1, p2):
    return np.sqrt(np.square(p2[0] - p1[0]) + np.square(p2[1] - p1[1]))


def points_to_vec(p1, p2):
    """ Returns the vector p1 -> p2. """
    v = np.empty(shape=2)
    v[0] = p2[0] - p1[0]
    v[1] = p2[1] - p1[1]
    return v


def vecs_angle(u, v):
    """ Computes cos(theta) = (u . v) / (|u| * |v|). """
    cos_ang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.arccos(cos_ang)


def kps_center(kps):
    """ Estimates the center of an image with the mean of the kps. """
    mean = np.zeros(2)
    for kp in kps:
        mean[0] += kp.pt[0]
        mean[1] += kp.pt[1]
    mean[0] /= len(kps)
    mean[1] /= len(kps)
    return mean


def get_kp_angle(kp1, kp2, center1, center2):
    vecq = points_to_vec(center1, kp1.pt)
    vect = points_to_vec(center2, kp2.pt)
    rtn = np.rad2deg(vecs_angle(vecq, vect))
    # diff = 135
    # if rtn > diff:
    #     print(f'angle between:\n\t{center1}->{kp1.pt}={vecq} and\n\t{center2}->{kp2.pt}={vect}\n\t= {rtn} degrees')
    return rtn


def find_kps_dist(center, kps):
    scale = np.empty(shape=len(kps))
    i = 0
    for kp in kps:
        scale[i] = eucl_distance(center, kp.pt)
        i += 1
    return scale  # maintains original kps order


def norm_angles(angles):
    for i in range(len(angles)):
        # if angles[i] < 0.0:
            # angles[i] = -angles[i]
        if angles[i] > 180.0:
            angles[i] = 360.0 - angles[i]
