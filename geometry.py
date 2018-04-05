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


def find_kp_angles(center, kps):
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
  return angles  # maintains original kps order


def find_kps_dist(center, kps):
  scale = np.empty(shape=len(kps))
  i = 0
  for kp in kps:
    scale[i] = eucl_distance(center, kp.pt)
    i += 1
  return scale  # maintains original kps order


def norm_angles(angles):
  for i in range(len(angles)):
    if angles[i] > 180.0:
      angles[i] = 360.0 - angles[i]
