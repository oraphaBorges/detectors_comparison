import cv2
import numpy as np
from matplotlib import pyplot as plt

NUM_OF_PAIRS = 1

def main():
    # Initiate detectors
    SIFT = cv2.xfeatures2d.SIFT_create()
    SURF = cv2.xfeatures2d.SURF_create()
    ORB = cv2.ORB.create()
    AKAZE = cv2.AKAZE.create()
    BRISK = cv2.BRISK.create()

    methods = {
        'SIFT': SIFT,
        'SURF': SURF,
        'ORB': ORB,
        'AKAZE': AKAZE,
        'BRISK': BRISK
    }

    cases = [
        'Same Object, Same Scale, Same Resolution, Indifferent POV',
        'Same Object, Different Scale, Same Resolution, Indifferent POV'
    ]

    for case in cases:
      print(case)
      for pair in range(NUM_OF_PAIRS):
        print('Pair {}/{}'.format(pair, NUM_OF_PAIRS))
        for name, method in methods.items():
          print(name)
          img1 = cv2.imread('photos/{}/{}a.jpg'.format(case,pair),0)
          img2 = cv2.imread('photos/{}/{}b.jpg'.format(case,pair),0)
          # find the keypoints and descriptors with ORB
          kp1, des1 = method.detectAndCompute(img1, None)
          kp2, des2 = method.detectAndCompute(img2, None)

          # create BFMatcher object
          bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

          # Match descriptors. (query,train)
          matches = bf.match(des1, des2)

          # Sort them in the order of their distance.
          matches = sorted(matches, key=lambda x: x.distance)

          img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches, outImg = None)

          cv2.imwrite('{}-{}.jpg'.format(name,case),img3)


if(__name__ == '__main__'):
    main()
