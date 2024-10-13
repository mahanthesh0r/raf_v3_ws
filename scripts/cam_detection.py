#!/usr/bin/env python3
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

# Initialize CvBridge
bridge = CvBridge()

class CamDetection:
    def __init__(self):
        self.pixel_selector = PixelSelector()
        self.tf_utils = raf_utils.TFUtils()
        self.inference_server = BiteAcquisitionInference()
        self.camera = RealSenseROS()

    
    def clear_plate(self):
        #items = self.inference_server.recognize_items(color_image)
        items = ['carrot','celery']

        self.inference_server.FOOD_CLASSES = items

        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        if camera_color_data is None:
            print("No camera data")
            return
        vis = camera_color_data.copy()

        annotated_image, detections, item_masks, item_portions, item_labels = self.inference_server.detect_items(camera_color_data)
        cv2.imshow('vis', annotated_image)
        cv2.waitKey(0)

        k = input("Are the detected items correct? (y/n): ")
        while k not in ['y', 'n']:
             k = ('Are the detected items correct? (y/n): ')
             if k == 'e':
                  exit(1)
        while k == 'n':
             exit(1)
        cv2.destroyAllWindows()

        clean_item_labels, _ = self.inference_server.clean_labels(item_labels)
        print("----- Clean Item Labels:", clean_item_labels)

        categories = self.inference_server.categorize_items(clean_item_labels)

        print("--------------------")
        print("Labels:", item_labels)
        print("Categories:", categories)
        print("Portions:", item_portions)
        print("--------------------")

        category_list = []
        labels_list = []
        per_food_masks = []
        per_food_portions = []

        for i in range(len(categories)):
            if labels_list.count(clean_item_labels[i]) == 0:
                    category_list.append(categories[i])
                    labels_list.append(clean_item_labels[i])
                    per_food_masks.append([item_masks[i]])
                    per_food_portions.append(item_portions[i])
            else:
                    index = labels_list.index(clean_item_labels[i])
                    per_food_masks[index].append(item_masks[i])
                    per_food_portions[index] += item_portions[i]

        print("Category List:", category_list)
        print("Labels List:", labels_list)
        print("Per Food Masks Len:", [len(x) for x in per_food_masks])
        print("Per Food Portions:", per_food_portions)
        
        self.inference_server.get_autonomous_action(annotated_image, camera_color_data, per_food_masks, category_list, labels_list, per_food_portions)

def main():
    rospy.init_node('cam_detection', anonymous=True)
    cd = CamDetection()
    cd.clear_plate()
    # camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
    # cd.grasping_pretzels(camera_color_data, camera_depth_data, camera_info_data, isOpenCv=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:    
        cv2.destroyAllWindows()


    

if __name__ == '__main__':
    main()
