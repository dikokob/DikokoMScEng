from __future__ import print_function
import cv2
import pandas as pd
import numpy as np
import math
import time
import os
from matplotlib import pyplot as plt


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
MIN_MATCH_COUNT = 10
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
          "TurtleMaps/Diff_Resolution/FullPath/FuulPath_res0.05.pgm"]


def matcher(img1, kp1, des1, img2, kp2, des2, detector):
    total = 0

    # BFMatcher with default params
    bf = cv2.BFMatcher(detector['normType'], crossCheck=True)

    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    for m in matches:
        total += m.distance

    average_distance = total/len(matches)

    # Draw first 10 matches.
    im_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    return im_matches, matches, average_distance


def transform(good_matches, img1, kp1, img2, kp2):

    im1_reg_final = None
    theta_recovered = None
    x_recovered = None
    y_recovered = None
    scale_recovered = None
    im1_reg = None

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

        H_affine, mask_affine = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC,
                                                            ransacReprojThreshold=3.0, maxIters=100000, confidence=0.99)

        ss = H_affine[0, 1]
        sc = H_affine[0, 0]
        scale_recovered = math.sqrt(ss * ss + sc * sc)
        theta_recovered = math.atan2(ss, sc) * 180 / math.pi
        x_recovered = H_affine[0][2]
        y_recovered = H_affine[1][2]

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

    return im1_reg_final, theta_recovered, im1_reg, scale_recovered

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

    return outliers_removed


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

    # Draw first 10 matches.
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

    DISTANCE_outliers_removed = outliers(DISTANCE)

    average_distance = np.mean(DISTANCE)

    average_distance_outliers_removed = np.mean(DISTANCE_outliers_removed)

    if average_distance_outliers_removed < 1:
        merge_successful = True
        print('successful merge')
    else:
        print('unsuccessful merge')

    return average_distance_outliers_removed, merge_successful, im_matches

values = []
full_start = time.time()
for detector in feature_detectors:

    for img in images:

        print("Reading reference image : ", img)
        imReference = cv2.imread(img, 0)

        scale_list = [0.2, 0.4, 0.6, 0.8]

        for scale in scale_list:

            for i in range(5):

                start = time.time()
                height, width = imReference.shape
                #M = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), theta, 1)
                width_new = int(imReference.shape[1] * scale)
                height_new = int(imReference.shape[0] * scale)
                dim = (width_new, height_new)
                # resize image
                imTransformed = cv2.resize(imReference, dim, interpolation=cv2.INTER_AREA)

                # find the keypoints and descriptors with SIFT
                kp_1, des_1 = detector['detector'].detectAndCompute(imReference, None)
                kp_2, des_2 = detector['detector'].detectAndCompute(imTransformed, None)

                # BFMatcher
                img3, good_matches_found, average = matcher(imReference, kp_1, des_1, imTransformed, kp_2, des_2, detector)
                im1Reg_final, thetaRecovered, imReg, scale_recovered= \
                    transform(good_matches_found, imReference, kp_1, imTransformed, kp_2)

                path = os.getcwd() + "\FeatureDetectionScale\{0}\{1}\{2}".format(detector['detector_name'],
                                                                                       str(scale), str(i))

                try:
                    os.makedirs(path)
                except:
                    pass

                cv2.imwrite('{0}/Matches_{1}Translation_{2}Detector.jpg'.
                            format(path, scale, detector['detector_name']), img3)
                cv2.imwrite('{0}/Transform{1}Translation_{2}Detector.jpg'.
                            format(path, scale, detector['detector_name']), im1Reg_final)
                cv2.imwrite('{0}/imTransformed{1}Translation_{2}Detector.jpg'.
                            format(path, scale, detector['detector_name']), imTransformed)
                cv2.imwrite('{0}/imReg{1}Translation_{2}Detector.jpg'.
                            format(path, scale, detector['detector_name']), imReg)

                average_distance, successful, im_verification = verification(imReg, imTransformed, detector)

                cv2.imwrite('{0}/verification_matches_{1}Translation_{2}Detector_'
                            '{3}Success.jpg'.format(path, scale, detector['detector_name'], successful), im_verification)

                end = time.time()
                elapsed_time = end - start

                values.append([img.split('/')[3], i, 0, thetaRecovered, average, len(kp_1), len(kp_2),
                               len(good_matches_found), detector['detector_name'], elapsed_time, successful,
                               average_distance, scale, scale_recovered])

                print('File: {}; Scale: {}; Scale_recovered: {}; Detector: {}; Compute_Time: {}'.format
                      (img.split('/')[3], round(scale), round(scale_recovered), detector['detector_name'], elapsed_time))

df = pd.DataFrame.from_records(values, columns=['file_name', 'iteration', 'actual_theta(degrees)',
                                                'calculated_theta(degrees)', 'average_distance',
                                                'original_num_features', 'rotated_num_features',
                                                'num_good_matches', 'detector_type', 'elapsed_time(s)',
                                                'successful_merge', 'distance_from_feature_after_merge',
                                                'scale', 'scale_recovered'])

df.to_csv('FeatureDetectionScale/result_translation.csv')
full_end = time.time()

print('Done, total time taken {}s'.format(full_end-full_start))
