from __future__ import print_function
import cv2
import pandas as pd
import numpy as np
import math
import time
import os, sys
from matplotlib import pyplot as plt

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
MIN_MATCH_COUNT = 0
sift = cv2.xfeatures2d.SIFT_create()

sift_detector = {'detector_name':'sift',
                 'detector': cv2.xfeatures2d.SIFT_create(),
                 'normType': cv2.NORM_L1 }

surf_detector = {'detector_name':'surf',
                 'detector': cv2.xfeatures2d.SURF_create(),
                 'normType': cv2.NORM_L1 }

orb_detector = {'detector_name':'orb',
                'detector': cv2.ORB_create(),
                'normType': cv2.NORM_HAMMING2 }

feature_detectors = [sift_detector, surf_detector, orb_detector]


# Read reference image
images = ["TurtleMaps/Diff_Resolution/2RobotsFull_R1/2RobotsFull_R1_res0.05.pgm",
          "TurtleMaps/Diff_Resolution/2RobotsFull_R2/2RobotsFull_R2_res0.05.pgm",
          "TurtleMaps/Diff_Resolution/MiniOverlap_R1/MiniOverlap_R1_res0.05.pgm",
          "TurtleMaps/Diff_Resolution/MiniOverlap_R2/MiniOverlap_R2_res0.05.pgm",
          "TurtleMaps/Diff_Resolution/FullPath/FuulPath_res0.05.pgm",
          "TurtleMaps/Diff_Resolution/Block_R1/Block_R1_res0.05.pgm",
          "TurtleMaps/Diff_Resolution/Block_R2/Block_R2_res0.05.pgm",
          "TurtleMaps/Diff_Resolution/SouthWing/SouthWing_res0.05.pgm"]

combinations = [[1,0],[0,2],[0,3],[0,4],[5,0],[0,6],[7,0],[1,2],[1,3],[1,4],[5,1],[1,6],[1,7],[2,3],[2,4],[2,5],
                [2,6],[2,7],[3,4],[3,5],[3,6],[3,7],[4,5],[4,6],[4,7],[5,6],[5,7],[6,7]]


def matcher(img1, kp1, des1, img2, kp2, des2, detector):
    total = 0

    # BFMatcher with default params
    bf = cv2.BFMatcher(detector['normType'], crossCheck=True)

    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Need to draw only good matches, so create a mask
    matchesMask = [0 for i in range(len(matches))]

    for i in range(int(len(matches)*0.1)):
        matchesMask[i] = 1

    for m in matches:
        total += m.distance

    average_distance = total/len(matches)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    # Draw first 10 matches.
    im_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

    return im_matches, matches, average_distance


def transform(good_matches, img1, kp1, img2, kp2, ransacReprojThreshold):

    im1_reg_final = None
    theta_recovered = None
    im1_reg = None
    im_inliers = None

    if len(good_matches) > MIN_MATCH_COUNT:

        # Extract location of good matches
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        # h, mask = cv2.findHomography(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=10, mask=None,
        #                             maxIters=100000, confidence=0.99)

        # Find estimateAffinePartial2D
        H_affine, mask_affine = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC,
                                                            ransacReprojThreshold=ransacReprojThreshold, maxIters=100000, confidence=0.99)


        matchesMask = mask_affine.ravel().tolist()

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Draw matches.
        im_inliers = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

        cv2.imwrite('Test.jpg',im_inliers)

        ss = H_affine[0, 1]  # [0,1] negative and [1,0] positive
        sc = H_affine[0, 0]
        scale_recovered = math.sqrt(ss * ss + sc * sc)
        theta_recovered = math.atan2(ss, sc) * 180 / math.pi

        print("MAP: Calculated scale difference: %.2f, "
              "Calculated rotation difference: %.2f" %
              (scale_recovered, theta_recovered))

        # Use homograph
        height, width = img2.shape
        #M = np.float32([h[0, 0:3],h[1, 0:3]])
        im1_reg = cv2.warpAffine(img1, H_affine, (width, height))

        # Merge
        im1_reg_sum = np.add(im1_reg.astype(int), img2.astype(int))
        im1_reg_final = (im1_reg_sum) / 2

        print('Image has been transformed')

    else:
        print
        "Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT)
        matchesMask = None

    return im1_reg_final, theta_recovered, im1_reg, scale_recovered, im_inliers


def outliers(data):
    # identify outliers with interquartile range
    from numpy.random import seed
    from numpy.random import randn
    from numpy import percentile

    # calculate interquartile range
    q25, q75 = percentile(data, 25), percentile(data, 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    # identify outliers
    outliers = [x for x in data if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    # remove outliers
    outliers_removed = [x for x in data if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))

    return outliers_removed, outliers


def verification(img1, img2, detector):
    merge_successful = False

    kp_1, des_1 = detector['detector'].detectAndCompute(img1, None)
    kp_2, des_2 = detector['detector'].detectAndCompute(img2, None)
    total = 0

    # BFMatcher with default params
    bf = cv2.BFMatcher(detector['normType'], crossCheck=True)

    matches = bf.match(des_1, des_2)

    # Sort them in the order of their distance.
    good_matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches.
    im_matches = cv2.drawMatches(img1, kp_1, img2, kp_2, matches, None, flags=2)


    # Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = kp_1[match.queryIdx].pt
        points2[i, :] = kp_2[match.trainIdx].pt

    distances = None

    dist_x_pow = np.power(points1[:, 0] - points2[:, 0], 2)
    dist_y_pow = np.power(points1[:, 1] - points2[:, 1], 2)
    DISTANCE = np.power(dist_x_pow + dist_y_pow, 0.5)

    DISTANCE_inliers, DISTANCE_outliers = outliers(DISTANCE)

    percentage_outliers = 100 * (len(DISTANCE_outliers)/len(DISTANCE))

    average_distance_outliers_removed = np.mean(DISTANCE_inliers)

    bf = cv2.BFMatcher()

    matches_inlier_len = bf.knnMatch(des_1, des_2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches_inlier_len:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatches(img1, kp_1, img2, kp_2, matches, None, flags=2)

    if average_distance_outliers_removed < 1:
        merge_successful = True
        print('successful merge')
    else:
        print('unsuccessful merge')

    return average_distance_outliers_removed, merge_successful, im_matches, percentage_outliers,img3

values = []
full_start = time.time()

detector = feature_detectors[0]
for ransacReprojThreshold in range(0,11):
    for i in range(len(combinations)):

        im_1_filename = images[combinations[i][0]]
        print("Reading im_1 : ", im_1_filename)
        im_1 = cv2.imread(im_1_filename, 0)


        im_2_filename = images[combinations[i][1]]
        print("Reading im_1 : ", im_2_filename)
        im_2 = cv2.imread(im_2_filename, 0)


        start = time.time()
        # find the keypoints and descriptors with SIFT
        kp_1, des_1 = detector['detector'].detectAndCompute(im_1, None)
        kp_2, des_2 = detector['detector'].detectAndCompute(im_2, None)

        # BFMatcher
        img3, good_matches_found, average = matcher(im_1, kp_1, des_1, im_2, kp_2, des_2, detector)
        im1Reg_final, thetaRecovered, imReg, scaleRecovered, im_inliers = transform(good_matches_found, im_1, kp_1, im_2, kp_2,
                                                                        ransacReprojThreshold)

        path = os.getcwd() + "\ImageMergeOnly\{0}\{1}".format(str(ransacReprojThreshold),str(i))
        os.makedirs(path)

        cv2.imwrite('{0}/Matches_{1}Detector.jpg'.
                    format(path, detector['detector_name']), img3)

        cv2.imwrite('{0}/Inliers_Matches_{1}Detector.jpg'.
                    format(path, detector['detector_name']), im_inliers)

        cv2.imwrite('ImageMergeOnly/{0}/Output_{1}Detector_Combination_{2}.jpg'.
                    format(str(ransacReprojThreshold), detector['detector_name'], i), im1Reg_final)

        cv2.imwrite('{0}/Output_{1}Detector.jpg'.
                    format(path, detector['detector_name']), im1Reg_final)

        cv2.imwrite('{0}/im_1_{1}Detector.jpg'.
                    format(path, detector['detector_name']), im_1)

        im_1_keypoints = im_1
        im_1_keypoints = cv2.drawKeypoints(im_1, kp_1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,outImage = im_1_keypoints)
        cv2.imwrite('{0}/im_1_keypoints_{1}Detector.jpg'.
                    format(path, detector['detector_name']), im_1_keypoints)

        cv2.imwrite('{0}/im_2{1}Detector.jpg'.
                    format(path, detector['detector_name']), im_2)
        im_2_keypoints = im_2
        im_2_keypoints = cv2.drawKeypoints(im_2, kp_2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage = im_2_keypoints)
        cv2.imwrite('{0}/im_2_keypoints_{1}Detector.jpg'.
                    format(path, detector['detector_name']), im_2_keypoints)

        cv2.imwrite('{0}/imReg{1}Detector.jpg'.
                    format(path, detector['detector_name']), imReg)

        average_distance, successful, im_verification, percentage_outliers, im_verification_inliers = verification(imReg, im_2, detector)

        cv2.imwrite('{0}/verification_matches_{1}Detector_'
                    '{2}Success.jpg'.format(path, detector['detector_name'], successful), im_verification)

        cv2.imwrite('{0}/verification_matches_inliers_{1}Detector_'
                    '{2}Success.jpg'.format(path, detector['detector_name'], successful), im_verification_inliers)

        end = time.time()
        elapsed_time = end - start

        values.append([im_1_filename.split('/')[3]+"_"+ im_2_filename.split('/')[3], ransacReprojThreshold, i, thetaRecovered, average, len(kp_1), len(kp_2),
                       len(good_matches_found), detector['detector_name'], elapsed_time, successful,
                       average_distance, percentage_outliers, scaleRecovered])

        print('File: {}; combination: {}; Calculated_Angle: {}; Detector: {}; Compute_Time: {}'.format
              (im_1_filename.split('/')[3]+"_"+ im_2_filename.split('/')[3], round(i), round(thetaRecovered),
               detector['detector_name'], elapsed_time))

df = pd.DataFrame.from_records(values, columns=['FileNames', 'ransacReprojThreshold', 'combinations',
                                                'calculated_theta(degrees)', 'average_distance',
                                                'original_num_features', 'rotated_num_features',
                                                'num_good_matches', 'detector_type', 'elapsed_time(s)',
                                                'successful_merge','distance_from_feature_after_merge',
                                                'percentage_outliers', 'calculated_scale'])

df.to_csv('ImageMergeOnly/result_translation.csv')
full_end = time.time()

print('Done, total time taken {}s'.format(full_end-full_start))
