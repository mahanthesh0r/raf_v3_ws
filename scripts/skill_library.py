import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import math
import threading
import raf_utils as utils
import cmath
import yaml

from robot_controller.robot_controller import KinovaRobotController

# ros imports
import rospy
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from rs_ros import RealSenseROS


class SkillLibrary:
    def __init__(self):

        self.tf_utils = utils.TFUtils()
        
        self.robot_controller = KinovaRobotController()

        


    def transfer_to_mouth(self, OFFSET = 0.1):

        self.robot_controller.move_to_transfer_pose()
        inp = input('Detect mouth center? (y/n): ')
        while inp != 'y':
            inp = input('Detect mouth center? (y/n): ')

        # check if mouth is open
        while True:
            mouth_open = rospy.wait_for_message('/mouth_open', Bool)
            if mouth_open.data:
                break
            else:
                print('Mouth is closed.')
    
        mouth_center_3d_msg = rospy.wait_for_message('/mouth_center', Point)
        mouth_center_3d = np.array([mouth_center_3d_msg.x, mouth_center_3d_msg.y, mouth_center_3d_msg.z])

        input("Press ENTER to move in front of mouth.")

        # create frame at mouth center
        mouth_center_transform = np.eye(4)
        mouth_center_transform[:3,3] = mouth_center_3d

        mouth_center_transform = self.tf_utils.getTransformationFromTF('base_link', 'camera_link') @ mouth_center_transform

        mouth_offset = np.eye(4)
        mouth_offset[2,3] = -OFFSET
        
        transfer_target = mouth_center_transform @ mouth_offset

        base_to_tooltip = self.tf_utils.getTransformationFromTF('base_link', 'tool_frame')

        transfer_target[:3,:3] = base_to_tooltip[:3,:3]

        pose = self.tf_utils.get_pose_msg_from_transform(transfer_target)

        # visualize on rviz
        self.tf_utils.publishTransformationToTF('base_link', 'mouth_center_transform', mouth_center_transform)
        self.tf_utils.publishTransformationToTF('base_link', 'transfer_target', transfer_target)

        print(f"Moving to transfer target: {pose}")

        self.robot_controller.move_to_pose(pose)

        input("Press ENTER to move back to before transfer pose.")
        self.robot_controller.move_to_transfer_pose()


    def getGripperWidth(self, width_point1, width_point2, finger_offset=0.6, pad_offset=0.35, insurance=0.975, close=1.175):
        #width of item in cm
        width =  np.linalg.norm(width_point1 - width_point2)
        # function transforming width to a gripper value
        grip_val = -7*((width*100)+(2*(finger_offset+pad_offset))) + 100
        grip_val = grip_val*insurance

        # make sure it doesn't exceed kinova limits
        if grip_val > 100 - 2*(finger_offset+pad_offset):
            grip_val = 100 - 2*(finger_offset+pad_offset)
        elif grip_val < 0:
            grip_val = 0
        
        grip_val = round(grip_val)/100

        return grip_val * close



   


