#!/usr/bin/env python3
import sys
import rospy
import os
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import tf
import tf.transformations as tf_trans
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from math import sqrt, inf, degrees, radians
from rs_ros import RealSenseROS
from pixel_selector import PixelSelector
import raf_utils
from cv_bridge import CvBridge
import math
from scipy.spatial.transform import Rotation
from inference_class import BiteAcquisitionInference
from robot_controller.robot_controller import KinovaRobotController
from skill_library import SkillLibrary

# Initialize CvBridge
bridge = CvBridge()

class CamDetection:
    def __init__(self):
        self.pixel_selector = PixelSelector()
        self.tf_utils = raf_utils.TFUtils()
        self.inference_server = BiteAcquisitionInference()
        self.camera = RealSenseROS()
        self.robot_controller = KinovaRobotController()
        self.skill_library = SkillLibrary()


    def mouth_transfer(self):
        self.skill_library.transfer_to_mouth() 

    def feeding(self):
        self.robot_controller.reset()
        self.inference_server.clear_plate()

    def cup_joint(self):
        self.robot_controller.move_to_cup_joint()
 
def main():
    rospy.init_node('cam_detection', anonymous=True)
    cd = CamDetection()
    cd.feeding()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:    
        cv2.destroyAllWindows()


    

if __name__ == '__main__':
    main()
