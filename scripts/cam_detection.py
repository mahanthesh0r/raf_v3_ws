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
import raf_utils


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
          # Create a stack to store commands
        self.command_stack = []

         # Subscribe to the speech_commands topic
        rospy.Subscriber('speech_commands', String, self.command_callback)


    def command_callback(self, msg):
        """
        Callback function to handle incoming commands.
        msg: The message received from the speech_commands topic.
        """
        command = msg.data
        rospy.loginfo("Received command: %s", command)
        

        # Push the command to the stack
        self.command_stack.append(command)
        

    def get_command(self):
        """
        Method to get the current command stack.
        """
        if self.command_stack:
            return self.command_stack[-1]
        else:
            return None
    
    def clear_stack(self):
        """
        Method to clear the command stack.
        """
        self.command_stack.clear()
        rospy.loginfo("Command stack cleared.")



    def mouth_transfer(self):
        self.skill_library.transfer_to_mouth() 

    def feeding(self):
        self.robot_controller.reset()
        self.inference_server.clear_plate()
    
    def drinking(self):
        self.robot_controller.move_to_cup_joint()
        rospy.sleep(4)
        self.inference_server.clear_plate(cup=True)
    
    def cup_joint(self):
        self.robot_controller.move_to_cup_joint()

    def camera_visualize(self):
        while True:
            camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
            if camera_color_data is None:
                print("No camera data")
                return
            vis = camera_color_data.copy()
            cv2.imshow('vis', vis)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

 
def main():
    rospy.init_node('cam_detection', anonymous=True)
    cd = CamDetection()
    raf_utils.play_sound("intro")
    # while True:
        
    #     if cd.get_command() == 'stop':
    #         cd.clear_stack()
    #         sys.exit(1)
    #         return
    #     elif cd.get_command() == 'feed':
    #         print("Feeding")
    #         cd.clear_stack()
    #         cd.feeding()
    #     elif cd.get_command() == 'drink':
    #         cd.clear_stack()
    #         cd.drinking()

        
    #cd.drinking()
    cd.feeding()
    #cd.camera_visualize()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:    
        cv2.destroyAllWindows()


    

if __name__ == '__main__':
    main()
