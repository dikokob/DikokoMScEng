from __future__ import print_function
import cv2
import pandas as pd
import numpy as np
import math
import time
import os, sys
from matplotlib import pyplot as plt
import itertools
import yaml
import glob
import rospy


def num2words(num):
    under_20 = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven',
                'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
    tens = ['Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
    above_100 = {100: 'Hundred', 1000: 'Thousand', 1000000: 'Million', 1000000000: 'Billion'}

    if num < 20:
        return under_20[num]

    if num < 100:
        return tens[(int)(num / 10) - 2] + ('' if num % 10 == 0 else ' ' + under_20[num % 10])

    # find the appropriate pivot - 'Million' in 3,603,550, or 'Thousand' in 603,550
    pivot = max([key for key in above_100.keys() if key <= num])

    return num2words((int)(num / pivot)) + ' ' + above_100[pivot] + (
        '' if num % pivot == 0 else ' ' + num2words(num % pivot))


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
MIN_MATCH_COUNT = 0
sift = cv2.xfeatures2d.SIFT_create()

sift_detector = {'detector_name': 'sift',
                 'detector': cv2.xfeatures2d.SIFT_create(),
                 'normType': cv2.NORM_L1}

surf_detector = {'detector_name': 'surf',
                 'detector': cv2.xfeatures2d.SURF_create(),
                 'normType': cv2.NORM_L1}

orb_detector = {'detector_name': 'orb',
                'detector': cv2.ORB_create(),
                'normType': cv2.NORM_HAMMING2}

feature_detectors = [sift_detector, surf_detector, orb_detector]

# Read reference image

images_maze = {'name': 'Maze', 'images': glob.glob('inputmaps/maze/*.pgm')}

images = [images_maze]


def matcher(des1, des2, detector):
    # BFMatcher with default params
    bf = cv2.BFMatcher(detector['normType'], crossCheck=True)

    matches = bf.match(des2, des1)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def transform(good_matches, img1, kp1, img2, kp2, ransacReprojThreshold, confidence):
    theta_recovered = None
    scale_recovered = None
    im2_transformed = None
    im_inliers = None

    # Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.trainIdx].pt
        points2[i, :] = kp2[match.queryIdx].pt

    # compute match ratio
    match_ratio = len(good_matches) / min(len(kp1), len(kp2))

    #  compute maximum iterations needed to estimate affine transformation matrix using RANSAC 
    maxIters = int(math.log(1 - confidence) / math.log(1 - math.pow(match_ratio,2)))

    # Find estimateAffinePartial2D
    H_affine, mask_affine = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC,
                                                        ransacReprojThreshold=ransacReprojThreshold,
                                                        maxIters=maxIters, confidence=confidence)

    # Mask Matches
    matchesMask = mask_affine.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Calculate Scale and Angle of rotation
    ss = H_affine[0, 1]  # [0,1] negative and [1,0] positive
    sc = H_affine[0, 0]
    scale_recovered = math.sqrt(ss * ss + sc * sc)
    theta_recovered = math.atan2(ss, sc) * 180 / math.pi

    rospy.loginfo("MAP: Calculated scale difference: %.2f, "
                  "Calculated rotation difference: %.2f" %
                  (scale_recovered, theta_recovered))

    # Use Transformtion Matrix
    height, width = img1.shape
    im2_transformed = cv2.warpAffine(img2, H_affine, (width, height))

    rospy.loginfo('Image has been transformed')

    return theta_recovered, im2_transformed, scale_recovered

def main(maps, ransacReprojThreshold=9, confidence=0.99):

    len_maps = len(maps)

    full_start = time.time()
    merge_results_dict_list = []

    # Select feature detector
    detector = feature_detectors[0]

    try:

        # Create Result Folder
        try:
            path = os.getcwd() + "/Results"
            rospy.loginfo("Creating path: {0}".format(path))
            os.makedirs(path)
        except:
            rospy.logerr("Failed to create path: {0}".format(path))

        # Extract Map cv2 types
        maps_cv2 = []
        for map_filename in maps:
            maps_cv2.append(cv2.imread(map_filename, 0))

        # Extract keypoints and descriptors
        kp_list = []
        des_list = []
        for map_cv2 in maps_cv2:
            kp, des = detector['detector'].detectAndCompute(map_cv2, None)
            kp_list.append(kp)
            des_list.append(des)

        # Align maps
        maps_successfully_transformed = []

        for i in range(1, len(maps_cv2)):

            merge_results_dict = {}

            # Get Good Matches
            good_matches_found = matcher(des_list[0], des_list[i], detector)

            # Algin Map
            thetaRecovered, imReg, scaleRecovered = transform(good_matches_found,
                                                                          maps_cv2[0],
                                                                          kp_list[0],
                                                                          maps_cv2[i],
                                                                          kp_list[i],
                                                                          ransacReprojThreshold,
                                                                          confidence)

            # Get map meta data
            original_metadata = yaml.load(open(maps[0].replace('pgm', 'yaml')))
            transformed_metadata = yaml.load(open(maps[i].replace('pgm', 'yaml')))

            # Check if transformation is successful based on the confidence
            success = True
            if (original_metadata['resolution'] * (1 - confidence)) < abs(
                    original_metadata['resolution'] - (transformed_metadata['resolution'] / scaleRecovered)):

                success = False

            else:

                maps_successfully_transformed.append(imReg)

        global_map = maps_cv2[0]

        del maps_cv2

        # Merge Successfully merged maps
        for transformed_map in maps_successfully_transformed:
            global_map = np.where(global_map == 205, transformed_map, global_map)

        # Store global map
        if global_map is not None:
            cv2.imwrite('{0}/global_map.jpg'.format(path), global_map)

    except:
        rospy.logerr("Unexpected error:{}".format(sys.exc_info()[0]))


    full_end = time.time()

    rospy.loginfo('Done, total time taken {0}s'.format(full_end - full_start))


if __name__ == "__main__":
    maps = ["inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R1/2RobotsFull_R1_res0.05.pgm",
            "inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R2/2RobotsFull_R2_res0.05.pgm",
            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R1/MiniOverlap_R1_res0.05.pgm"]

    main(maps)
