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

def num2words(num):
    under_20 = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Eleven','Twelve','Thirteen','Fourteen','Fifteen','Sixteen','Seventeen','Eighteen','Nineteen']
    tens = ['Twenty','Thirty','Forty','Fifty','Sixty','Seventy','Eighty','Ninety']
    above_100 = {100: 'Hundred',1000:'Thousand', 1000000:'Million', 1000000000:'Billion'}
 
    if num < 20:
         return under_20[num]

    if num < 100:
        return tens[(int)(num/10)-2] + ('' if num%10==0 else ' ' + under_20[num%10])
 
    # find the appropriate pivot - 'Million' in 3,603,550, or 'Thousand' in 603,550
    pivot = max([key for key in above_100.keys() if key <= num])
 
    return num2words((int)(num/pivot)) + ' ' + above_100[pivot] + ('' if num%pivot==0 else ' ' + num2words(num%pivot))

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
images_turtle = {'name': 'Turtle',
                 'images': ["inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R1/2RobotsFull_R1_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R2/2RobotsFull_R2_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R1/MiniOverlap_R1_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R2/MiniOverlap_R2_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/FullPath/FuulPath_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/Block_R1/Block_R1_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/Block_R2/Block_R2_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/SouthWing/SouthWing_res0.05.pgm"]}

images_turtle_2robot = {'name': 'Turtle2Robots',
                        'images': ["inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R1/2RobotsFull_R1_res0.1.pgm",
                                    "inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R1/2RobotsFull_R1_res0.05.pgm"]}

images_turtle_diffRes = {'name': 'TurtleDiffRes',
                 'images': ["inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R1/2RobotsFull_R1_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R2/2RobotsFull_R2_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R1/MiniOverlap_R1_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R2/MiniOverlap_R2_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/FullPath/FuulPath_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/Block_R1/Block_R1_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/Block_R2/Block_R2_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/SouthWing/SouthWing_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R1/2RobotsFull_R1_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R2/2RobotsFull_R2_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R1/MiniOverlap_R1_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R2/MiniOverlap_R2_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/Block_R1/Block_R1_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/Block_R2/Block_R2_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/SouthWing/SouthWing_res0.1.pgm"]}

images_turtle_diffRes4 = {'name': 'TurtleDiffRes4',
                 'images': ["inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R1/2RobotsFull_R1_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R2/2RobotsFull_R2_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R1/MiniOverlap_R1_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R2/MiniOverlap_R2_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/FullPath/FuulPath_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/Block_R1/Block_R1_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/Block_R2/Block_R2_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/SouthWing/SouthWing_res0.05.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R1/2RobotsFull_R1_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/2RobotsFull_R2/2RobotsFull_R2_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R1/MiniOverlap_R1_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R2/MiniOverlap_R2_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/Block_R1/Block_R1_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/Block_R2/Block_R2_res0.1.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/SouthWing/SouthWing_res0.1.pgm",
                            "inputmaps/TurtleMaps/Slow/slow_res0.05.pgm",
                            "inputmaps/TurtleMaps/Slow/slow_res0.0125.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/MiniOverlap_R2/MiniOverlap_R2_res0.2.pgm",
                            "inputmaps/TurtleMaps/Diff_Resolution/Block_R2/Block_R2_res0.2.pgm"]}


images_maze = {'name': 'Maze','images': glob.glob('inputmaps/maze/*.pgm') }

images_maze_diff_scale = {'name': 'Maze_Diff_Scale','images': glob.glob('inputmaps/maze_diff_scale/*.pgm') }

images_experiment = {'name': 'Environment', 'images': ["inputmaps/experiment_1/maze_0.pgm", "inputmaps/experiment_1/maze_1.pgm",
                                                       "inputmaps/experiment_1/maze_2.pgm", "inputmaps/experiment_1/maze_3.pgm", 
                                                       "inputmaps/experiment_1/maze_4.pgm",  "inputmaps/experiment_1/maze_5.pgm",  
                                                       "inputmaps/experiment_1/maze_6.pgm", "inputmaps/experiment_1/maze_7.pgm",
                                                       "inputmaps/experiment_1/maze_8.pgm", "inputmaps/experiment_1/maze_9.pgm"]}

images_fr0 = {'name': 'fr0', 'images': ["inputmaps/fr0/fr0.pgm", "inputmaps/fr0/fr1.pgm", "inputmaps/fr0/fr2.pgm", "inputmaps/fr0/fr3.pgm", "inputmaps/fr0/fr4.pgm",
                                        "inputmaps/fr0/fr5.pgm", "inputmaps/fr0/fr6.pgm", "inputmaps/fr0/fr7.pgm", "inputmaps/fr0/fr8.pgm"]}                                               

images = [images_turtle_diffRes4]

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

    #im1_reg_final = None
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
        #im1_reg_sum = np.add(im1_reg.astype(int), img2.astype(int))
       # im1_reg_final = (im1_reg_sum) / 2

        print('Image has been transformed')

    else:
        print
        "Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT)
        matchesMask = None

    return  theta_recovered, im1_reg, scale_recovered, im_inliers



def main(num):
    

    # images = [images_turtle, images_turtle_2robot, images_turtle_diffRes, images_maze, images_experiment, images_fr0]
    
    num = num

    for image in images:
        combinations = itertools.combinations(list(range(len(image['images']))), num)
        element_list = [list(element) for element in combinations]
        combinations = {'num': num2words(num), 'combinations':  element_list}
        image['combinations'] = combinations

    

    for image in images:
        full_start = time.time()


        detector = feature_detectors[0]

        values = []
        bench_marking_main_list = []

        try:

            for combination in image['combinations']['combinations']:

                try:
                    path = os.getcwd() + "\\benchmarking\outputs\{0}Comb{1}FinalImageMerger\{2}".format(image['combinations']['num'], image['name'], combination)
                    os.makedirs(path)

                except:
                    pass

                img_filename_list = []
                img_list = []
                for img_index in combination:
                    img_filename_list.append(image['images'][img_index])
                    img_list.append(cv2.imread(image['images'][img_index], 0))

                kp_list = []
                des_list = []

                for img in img_list:
                    kp, des = detector['detector'].detectAndCompute(img, None)
                    kp_list.append(kp)
                    des_list.append(des)

                start = time.time()
                #goodmatches_list = []
                img_reg_list = []
                bench_marking_dict_list = []
                # BFMatcher
                for i in range(1, len(img_list)):
                    bench_marking_dict = {}
                    img3, good_matches_found, average = matcher(img_list[i], kp_list[i], des_list[i], img_list[0],
                                                                kp_list[0], des_list[0], detector)
                    #goodmatches_list.append(good_matches_found)

                    thetaRecovered, imReg, scaleRecovered, im_inliers = transform(good_matches_found, img_list[i],
                                                                                                    kp_list[i], img_list[0],
                                                                                                    kp_list[0], 9)

                    original = yaml.load(open(img_filename_list[0].replace('pgm', 'yaml')))
                    transformed = yaml.load(open(img_filename_list[i].replace('pgm', 'yaml')))
                    originalResolution_div_transformedResolution = (transformed['resolution']/original['resolution'])
                    originalResolution_div_transformedResolution_minus_scaleRecovered = abs((transformed['resolution']/original['resolution']) - (scaleRecovered))

                    bench_marking_dict['original_filename'] = img_filename_list[0]
                    bench_marking_dict['original_resolution'] = original['resolution']
                    bench_marking_dict['transformed_filename'] = img_filename_list[i]
                    bench_marking_dict['transformed_resolution'] = transformed['resolution']
                    bench_marking_dict['originalResolution_div_transformedResolution'] = originalResolution_div_transformedResolution
                    bench_marking_dict['originalResolution_div_transformedResolution_minus_scaleRecovered'] = originalResolution_div_transformedResolution_minus_scaleRecovered
                    bench_marking_dict['thetaRecovered'] = thetaRecovered
                    bench_marking_dict['scaleRecovered'] = scaleRecovered
                    bench_marking_dict['numGoodMatches'] = len(good_matches_found)
                    bench_marking_dict['combination'] = combination
                    bench_marking_dict_list.append(bench_marking_dict) 
                    bench_marking_main_list.append(bench_marking_dict)       

                    img_reg_list.append(imReg)                    
                        
                    cv2.imwrite('{0}/Matches_{1}.jpg'. format(path, combination[i]), im_inliers)
                    cv2.imwrite('{0}/OriginalImage_{1}.jpg'. format(path, combination[i]), img_list[i])
                    cv2.imwrite('{0}/TransformedImage_{1}.jpg'. format(path, combination[i]), imReg)
                    
                df_bench_marking = pd.DataFrame(bench_marking_dict_list)
                df_bench_marking.to_csv('{0}\\bench_marking.csv'. format(path))
                del df_bench_marking

                path = (path).replace('(fail)','')
                path = (path).replace('(success)','')

                final_image_1 = img_list[0]
                cv2.imwrite('{0}/OriginalImage.jpg'. format(path), img_list[0])
                del img_list
            
                i = 1
                for img_reg in img_reg_list:
                    final_image_1 = np.where(final_image_1 == 205, img_reg, final_image_1)
                    #final_image = np.add(final_image, img_reg)
                    i = i + 1
                
                if final_image_1 is not None:   
                    #cv2.imwrite('{0}/Final.jpg'. format(path), final_image)
                    cv2.imwrite('{0}/Final.jpg'. format(path), final_image_1)


                end = time.time()
                elapsed_time = end - start
        except:
            pass
            
        
        df_bench_marking = pd.DataFrame(bench_marking_main_list)
        df_bench_marking.to_csv('{0}\\bench_marking_final.csv'. format(path))

        full_end = time.time()

        print('Done, total time taken {}s'.format(full_end-full_start))

if __name__ == "__main__":

    nums = [2]

    for num in nums:
        main(num)
    pass
