import numpy as np
import math
import tf2_ros
import rospy
import cv2
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose, PoseStamped
from geometry_msgs.msg import Pose, TransformStamped
from math import atan2, cos, sin, sqrt, pi
import time
import pygame
import random
import tempfile
import os


def angle_between_pixels(source_px, target_px, image_width, image_height, orientation_symmetry = False):
    def angle_between(p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    if orientation_symmetry and source_px[1] > target_px[1]:
        source_px, target_px = target_px, source_px
    source_px_cartesian = np.array([source_px[0], image_height-source_px[1]])
    target_px_cartesian = np.array([target_px[0], image_height-target_px[1]])
    angle = angle_between(np.array([-image_width,0]), source_px_cartesian-target_px_cartesian)
    robot_angle_offset = -90
    return angle + robot_angle_offset

# function to get the points of the "top" and "bottom" of the box
def get_box_points(mask):
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)


    # get a rotated rectangle around the segmentation
    rect = cv2.minAreaRect(largest_contour)
    # get the box points of the rectangle and convert to integers
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    

    if np.linalg.norm(box[0]-box[1]) < np.linalg.norm(box[1]-box[2]):
        # then the longer side is the one between the first and second points
        # i want the midpoints between the widths
        p1 = (box[0] + box[1]) / 2
        p2 = (box[2] + box[3]) / 2
        
        # grab points for width calculation
        width_p1 = box[0]
        width_p2 = box[1]

        # get the length of the shorter side
    else:
        p1 = (box[1] + box[2]) / 2
        p2 = (box[3] + box[0]) / 2

        # grab points for width calculation
        width_p1 = box[1]
        width_p2 = box[2]
    
    # Convert midpoints to integers
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))

    return p1, p2, width_p1, width_p2, box

def get_width_points(grasp_point,mask):
    p1,p2,width_p1,width_p2,box = get_box_points(mask)
    # get distance from upper midpoint to grasp point
    #dist = np.linalg.norm(np.array(p1) - np.array(grasp_point))
    dist = tuple(a-b for a,b in zip(p1, grasp_point))
    # get locations of the grasp points on the rotated rectangle
    grasp_point1 = tuple(a-b for a,b in zip(width_p1, dist))
    grasp_point2 = tuple(a-b for a,b in zip(width_p2, dist))
    # draw the grasp line
    # cv2.line(mask, tuple(grasp_point1), tuple(grasp_point2), 255, 2)
    return grasp_point1, grasp_point2
    




# function to get the points of the "top" and "bottom" of the box
def get_cup_box_points(mask):
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)


    # get a rotated rectangle around the segmentation
    rect = cv2.minAreaRect(largest_contour)
    # get the box points of the rectangle and convert to integers
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    

    if np.linalg.norm(box[0]-box[1]) < np.linalg.norm(box[1]-box[2]):
        # then the longer side is the one between the first and second points
        # i want the midpoints between the widths
        p1 = (box[1] + box[2]) / 2
        p2 = (box[3] + box[0]) / 2
        
        # grab points for width calculation
        width_p1 = box[1]
        width_p2 = box[2]

        # get the length of the shorter side
    else:
        p1 = (box[0] + box[1]) / 2
        p2 = (box[2] + box[3]) / 2

        # grab points for width calculation
        width_p1 = box[0]
        width_p2 = box[1]
    
    # Convert midpoints to integers
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))

    return p1, p2, width_p1, width_p2, box

def get_cup_box_points_v2(mask):
     # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the centroid of the largest contour
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, None, None, None, None
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])

    # Find the farthest points to the right and left along the horizontal axis
    far_right = max(largest_contour, key=lambda point: point[0][0] if abs(point[0][1] - centroid_y) < 5 else -np.inf)
    far_left = min(largest_contour, key=lambda point: point[0][0] if abs(point[0][1] - centroid_y) < 5 else np.inf)

    # Convert points to tuples
    far_right_point = (far_right[0][0], centroid_y)
    far_left_point = (far_left[0][0], centroid_y)

    # Draw points on the mask
    mask_with_points = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.circle(mask_with_points, far_right_point, 5, (0, 0, 255), -1)  # Red point
    cv2.circle(mask_with_points, far_left_point, 5, (0, 0, 255), -1)   # Red point
    cv2.circle(mask_with_points, (centroid_x, centroid_y), 5, (255, 0, 0), -1)  # Blue point for centroid

    return far_right_point, far_left_point, (centroid_x, centroid_y), mask_with_points


# custom one to return angle between 0 and 180 degress
def pretzel_angle_between_pixels(center, lower):
    center_y = center[1]
    lower_y = lower[1]

    if center_y < lower_y:
        p2 = center
        p1 = lower
    else:
        p1 = center
        p2 = lower
    
    a = abs(p2[1] - p1[1])
    b = abs(p2[0] - p1[0])
    if b == 0:
        b=0.001
    
    angle = math.degrees(math.atan(a/b))

    if p1[0] > p2[0]:
        angle = 180 - angle

    return angle

def getOrientation(mask):
  contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  for i, c in enumerate(contours):
      
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(c)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = c[i,0,0]
        data_pts[i,1] = c[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]
    
    ## [visualization]
    # Draw the principal components
    
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    ## [visualization]
    
    # Label with the rotation angle
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    print(label)

 
  return angle


def pixel2World(camera_info, image_x, image_y, depth_image, box_width = 2):

    print("(image_y,image_x): ",image_y,image_x)
    print("depth image: ", depth_image.shape[0], depth_image.shape[1])

    if image_y >= depth_image.shape[0] or image_x >= depth_image.shape[1]:
        return False, None
    

    depth = depth_image[image_y, image_x]
    print("depth of item: ", depth)

    if math.isnan(depth) or depth < 0.05 or depth > 1.0:

        depth = []
        for i in range(-box_width,box_width):
            for j in range(-box_width,box_width):
                if image_y+i >= depth_image.shape[0] or image_x+j >= depth_image.shape[1]:
                    return False, None
                pixel_depth = depth_image[image_y+i, image_x+j]
                if not (math.isnan(pixel_depth) or pixel_depth < 50 or pixel_depth > 1000):
                    depth += [pixel_depth]

        if len(depth) == 0:
            print("No valid depth values found")
            return False, None

        depth = np.mean(np.array(depth))

    depth = depth/1000.0 # Convert from mm to m

    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]  

    # Convert to world space
    world_x = (depth / fx) * (image_x - cx)
    world_y = (depth / fy) * (image_y - cy)
    world_z = depth

    return True, np.array([world_x, world_y, world_z])


def world2Pixel(camera_info, world_x, world_y, world_z):

    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]  

    image_x = world_x * (fx / world_z) + cx
    image_y = world_y * (fy / world_z) + cy

    return image_x, image_y

def validate_with_user(question):
    user_input = input(question + "(y/n): ")
    while user_input != "y" and user_input != "n":
        user_input = input(question + "(y/n): ")
    if user_input == "y":
        return True
    else:
        return False
    
def get_category_from_label(food_classes):
    for food_class in food_classes:
        if food_class in ['egg rolls', 'chicken tenders', 'pretzel rods', 'carrots', 'celery','fried chicken']:
            return 'multi-bite'
        else:
            return 'single-bite'
        
def save_camera_data(camera_data):
    file = '/home/labuser/raf_v3_ws/src/raf_v3/scripts/img_data/'
    cv2.imwrite(file + "_camera_color_data.jpg", camera_data)
        
        
        
def randomize_selection(items):
    if not items:
        return None
    choice = random.choice(items)
    return choice if isinstance(choice, list) else [choice]

def find_gripper_values(food_item,table_depth,food_depth):
    print("food item: ", food_item[0])
    #['pretzel bites','celery', 'carrot','pretzel rods','sushi','green grapes','egg rolls','watermelon','chicken tenders', 
    # 'chocolate','pretzel rods','penne pasta','tomato','green vegetable','donut','chicken nugget']
    if food_item[0] in ['pretzel bites','chocolate','green grapes',' yellow popcorn']:
        # chicken nugget is really a vegan mandarin bite
        close = 1.085
        # height of the food
        height = 0.01
    elif food_item[0] in ['chicken tenders','egg rolls','sushi','fried chicken']:
        # these food items are taller and squishier
        close = 1.185
        height = 0.015
    elif food_item[0] in ['watermelon']:
        close = 1.13
    elif food_item[0] in ['donut']:
        close = 1.15
    elif food_item[0] in ['almonds','pretzel rods','gummy bears']:
        close = 1.05
        height = 0.005
    elif food_item[0] in ['single penne pasta']:
        close = 1.1
        height = 0.008
    elif food_item[0] in ['carrots']:
        close = 1.08
        height = 0.01
    elif food_item[0] in ['chicken nuggets']:
        close = 1.10
        height = 0.012
    else:
        close = 1.1
        height = 0.01

    # get the height of the food
    height = abs(table_depth - food_depth[2])
    print("Height of selected food item: ", height)

    print("close: ", close)
    #print("height: ", height)

    # gripper should go 78% of the height of the food item
    if food_item[0] in ['suhsi']:
        height = height*0.88
    else:
        height = height*0.78
        
    return close, height
        
        

def organize_food_data(categories, clean_item_labels, item_masks, item_portions):
    category_list, labels_list, per_food_masks, per_food_portions = [], [], [], []
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
    return category_list, labels_list, per_food_masks, per_food_portions   

def play_sound(type):
    pygame.mixer.init()
    if type == "intro":
        pygame.mixer.music.load("../assets/intro.mp3")
    elif type == "notification":
        pygame.mixer.music.load("../assets/notification.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue


def image_from_camera(camera_data):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_file_name = temp_file.name
        cv2.imwrite(temp_file_name, camera_data)
    return temp_file_name

def delete_temp_file(temp_file_name):
    os.remove(temp_file_name)


'''
Converts a list of items to a string that can be used in a prompt
'''
def list_to_prompt_string(item):
    if not item:
        return ""
    return item[0] + " ."
    
def getJointVelocity(current_joint, target_joint, duration):
    joint_velocities = []
    for i in range(len(current_joint)):
        error = min(abs(target_joint[i] - current_joint[i]), 360 - abs((target_joint[i] - current_joint[i])))
        
        if not (360+target_joint[i]==current_joint[i]+error) and (current_joint[i] > target_joint[i]):
            error = -error
            
        velocity = round(math.radians(error/duration), 8)
        joint_velocities.append(velocity)
        print("Joint velocity: ", velocity)
    return joint_velocities

class TFUtils:
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer() # Using default cache time of 10 secs
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.control_rate = rospy.Rate(100)
    
    def getTransformationFromTF(self, source_frame, target_frame):

        while not rospy.is_shutdown():
            try:
                # print(f"Looking for transform from {source_frame} to {target_frame} using tfBuffer.lookup_transform...")
                transform = self.tfBuffer.lookup_transform(source_frame, target_frame, rospy.Time())
                # print("Got transform!")
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.control_rate.sleep()
                continue

        T = np.zeros((4,4))
        T[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        T[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        T[3,3] = 1



        print("Translation: ", T[:3,3])
        print("Rotation in quaternion: ", transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w)
        print("Rotation in euler: ", Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_euler('xyz', degrees=True))
        

        return T
    
    def publishTransformationToTF(self, source_frame, target_frame, transform):

        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = source_frame
        t.child_frame_id = target_frame

        t.transform.translation.x = transform[0][3]
        t.transform.translation.y = transform[1][3]
        t.transform.translation.z = transform[2][3]

        R = Rotation.from_matrix(transform[:3,:3]).as_quat()
        t.transform.rotation.x = R[0]
        t.transform.rotation.y = R[1]
        t.transform.rotation.z = R[2]
        t.transform.rotation.w = R[3]

        self.broadcaster.sendTransform(t)

    def get_pose_msg_from_transform(self, transform):

        pose = Pose()
        pose.position.x = transform[0,3]
        pose.position.y = transform[1,3]
        pose.position.z = transform[2,3]

        quat = Rotation.from_matrix(transform[:3,:3]).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        return pose
