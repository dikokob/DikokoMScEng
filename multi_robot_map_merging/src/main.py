#! /usr/bin/env python

import rospy
import sys
import tf
import cv2
import math
import time
import datetime
import numpy as np
import os
import image_registration
import glob

from nav_msgs.msg import OccupancyGrid, Odometry, MapMetaData


class MapMerge:
    """
    Map Merging ROS
    """

    def __init__(self):
        """
        init class to initialise the class

        - rate default 10 (script run rate)

        - RANSAC parameters:
            - ransacReprojThreshold default 9 ()
            - confidence default 0.99 ()
        """
        rospy.init_node('Map Merging Node')

        self.ransacReprojThreshold = rospy.get_param("~ransacReprojThreshold", 9)
        self.confidence = rospy.get_param("~confidence", 0.99)

        # Script Run Rate
        self.rate = rospy.Rate(rospy.get_param("~rate", 10))

    def StartMapMerging(self):


        rospy.loginfo("start Map Merging Node")

        while not rospy.is_shutdown():

            # Get list of ROS topics
            list_of_topics = rospy.get_published_topics()
            rospy.loginfo("Got {} number of total topics".format(len(list_of_topics))) 

            # Look for topics of type nav_msgs/OccupancyGrid
            list_of_map_topics = []         
            for topic in list_of_topics:
                if topic[1] == 'nav_msgs/OccupancyGrid':
                    list_of_map_topics.append(topic)
            
            rospy.loginfo("Got {} number of map topics".format(len(list_of_map_topics)))

            # Create inputmaps folder
            try:
                path = os.getcwd() + "/inputmaps"
                rospy.loginfo("Creating path: {0}".format(path))
                os.makedirs(path)
            except:
                rospy.logerr("Failed to create path: {0}".format(path))

            rospy.loginfo("Save all the maps")

            # Save all maps using map_server
            for map_topic in list_of_map_topics:
                os.system("rosrun map_server map_saver map:=/{0} -f inputmaps/{1}".
                          format(map_topic[0], map_topic[0][1:].replace('/', '_')))

            list_of_maps = glob.glob('inputmaps/*.pgm')

            # Attempt a map merge
            if len(list_of_maps) != 0:
                # Run merging algorithm
                image_registration.main(list_of_maps, self.ransacReprojThreshold, self.confidence)

            self.rate.sleep()


if __name__ == "__main__":

    try:
        map_merger = MapMerge()
        map_merger.StartMapMerging()
    except rospy.ROSInterruptException:
        pass



            





