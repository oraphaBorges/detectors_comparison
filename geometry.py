from sympy import Point, Line
import numpy


# Finds the image's center
def image_center(image):
    return Point(image.shape[1] / 2, image.shape[0] / 2)


# Finds the Key's points Angles
def find_kp_angles(kp1, kp2, matches, center1, center2):
    central_line = Line(center1, center2.translate(2*center2.x))
    angles = []
    for match in matches:
        p1 = Point(kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1])
        p2 = Point(kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1])
        match_line = Line(p1, p2.translate(2*center2.x))
        angles.append(float(central_line.angle_between(match_line)))
    return angles


# Finds the Scale between images
def find_scale(kp1, kp2, matches, center1, center2):
    scale = 0.0
    for match in matches:
        p1 = Point(kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1])
        p2 = Point(kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1])
        d1 = center1.distance(p1)
        d2 = center2.distance(p2)
        scale += d1 / d2
    return scale


# Finds the Scale between image
def find_kp_distances(kps, center):
    distances = []
    for kp in kps:
        p = Point(kp.pt[0], kp.pt[1])
        d = center.distance(p)
        distances.append(float(d))
    return distances
