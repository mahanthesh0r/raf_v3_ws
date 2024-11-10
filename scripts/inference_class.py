#!/usr/bin/env python3
import os
import base64
import time
import requests
import sys
import ast
import cv2
import supervision as sv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor, Compose
from vision_utils import efficient_sam_box_prompt_segment, detect_plate, cleanup_mask, mask_weight, detect_centroid, detect_lower_center, get_grasp_points
import raf_utils
from rs_ros import RealSenseROS
from scipy.spatial.transform import Rotation
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_slerp
from math import sqrt, inf, degrees, radians
from robot_controller.robot_controller import KinovaRobotController
from pixel_selector import PixelSelector
import rospy




from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

#from skill_library import SkillLibrary


PATH_TO_GROUNDED_SAM = '/home/labuser/raf_v3_ws/Grounded-Segment-Anything'
PATH_TO_DEPTH_ANYTHING = '/home/labuser/raf_v3_ws/Depth-Anything'
USE_EFFICIENT_SAM = False

sys.path.append(PATH_TO_DEPTH_ANYTHING)

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

class GPT4VFoodIdentification:
    def __init__(self, api_key, prompt_dir):
        self.api_key = api_key

        self.PREFERENCE = "custom"
        self.history_file_path = "/home/labuser/raf_v3_ws/src/raf_v3/scripts"
        

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
            }
        self.prompt_dir = prompt_dir
        
        


        if self.PREFERENCE == "alternate":
            with open("%s/alternating_prompt.txt"%self.prompt_dir, 'r') as f:
                self.prompt_text = f.read()
                self.previous_bite = self.get_history_food()
                self.prompt_text = self.prompt_text.replace("{variable}", self.previous_bite)
        elif self.PREFERENCE == "carrots_first":
            with open("%s/carrots_first_prompt.txt"%self.prompt_dir, 'r') as f:
                self.prompt_text = f.read()
        else:
            with open("%s/custom.txt"%self.prompt_dir, 'r') as f:
                self.prompt_text = f.read()

    def encode_image(self, openCV_image):
        retval, buffer = cv2.imencode('.jpg', openCV_image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_history_food(self):
        with open("%s/history.txt"%self.history_file_path, 'r') as f:
            history = f.read()
            previous_bite = ast.literal_eval(history)
            return previous_bite[-1]
        
    def update_history(self, food):
        with open("%s/history.txt"%self.history_file_path, 'r') as f:
            history = f.read()
            previous_bite = ast.literal_eval(history)
            previous_bite.append(food)
            with open("%s/history.txt"%self.history_file_path, 'w') as f:
                f.write(str(previous_bite))
            
            
        
    def prompt(self, image):
        # Getting the base64 string
        base64_image = self.encode_image(image)

        payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": self.prompt_text
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        # grab and format response from API
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        response_text =  response.json()['choices'][0]["message"]["content"]
        return response_text
    

class BiteAcquisitionInference:
    def __init__(self):
        with torch.no_grad():
            torch.cuda.empty_cache()

        # what it was before
        # self.Z_OFFSET = 0.01
        # self.GRIPPER_OFFSET = 0.00 #0.07
        # self.CAMERA_OFFSET = 0.035

        self.Z_OFFSET = 0.01
        self.GRIPPER_OFFSET = 0.00 #0.07
        self.CAMERA_OFFSET = 0.042

        self.AUTONOMY = True
        
        
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.api_key = os.environ['OPENAI_API_KEY']
        self.gpt4v_client = GPT4VFoodIdentification(self.api_key, '/home/labuser/raf_v3_ws/src/raf_v3/scripts/prompts')

        self.FOOD_CLASSES = ["pretzel"]
        self.HOLDER_CLASSES = ["bowl"]
        # self.BOX_THRESHOLD = 0.036
        # self.TEXT_THRESHOLD = 0.028
        # self.NMS_THRESHOLD = 0.4

        # thresholds for pretzels
        self.BOX_THRESHOLD = 0.03
        self.TEXT_THRESHOLD = 0.02
        self.NMS_THRESHOLD = 0.4

        self.camera = RealSenseROS()
        self.tf_utils = raf_utils.TFUtils()
        self.robot_controller = KinovaRobotController()

        self.pixel_selector = PixelSelector()

        

         


        
        # Grounding DINO stuff
        self.GROUNDING_DINO_CONFIG_PATH = PATH_TO_GROUNDED_SAM + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.GROUNDING_DINO_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/groundingdino_swint_ogc.pth"

         # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=self.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH)
        self.use_efficient_sam = USE_EFFICIENT_SAM

        if self.use_efficient_sam:
            self.EFFICIENT_SAM_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/efficientsam_s_gpu.jit"
            self.efficientsam = torch.jit.load(self.EFFICIENT_SAM_CHECKPOINT_PATH)   

        else:
            # Segment-Anything checkpoint
            SAM_ENCODER_VERSION = "vit_h"
            SAM_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/sam_vit_h_4b8939.pth"

            # Building SAM Model and SAM Predictor
            sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
            sam.to(device=self.DEVICE)
            self.sam_predictor = SamPredictor(sam)

        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(self.DEVICE).eval()

        self.depth_anything_transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])


    def clear_plate(self):
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        self.robot_controller.reset()
        #items = self.recognize_items(camera_color_data)
        items = ['pretzel']

        self.FOOD_CLASSES = items

        if camera_color_data is None:
            print("No camera data")
            return
        
        annotated_image, detections, item_masks, item_portions, item_labels = self.detect_food(camera_color_data)
       
        if not self.AUTONOMY:
            vis = camera_color_data.copy()
            cv2.imshow('vis', annotated_image)
            cv2.waitKey(0)
            k = input("Are the detected items correct? (y/n): ")
            while k not in ['y', 'n']:
                k = ('Are the detected items correct? (y/n): ')
                if k == 'e':
                  sys.exit(1)
            while k == 'n':
                sys.exit(1)
            cv2.destroyAllWindows()

        clean_item_labels, _ = self.clean_labels(item_labels)
        print("----- Clean Item Labels:", clean_item_labels)

        categories = self.categorize_items(clean_item_labels)
        
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

        self.get_autonomous_action(annotated_image, camera_color_data, per_food_masks, category_list, labels_list, per_food_portions)


    def recognize_items(self, image):
        response = self.gpt4v_client.prompt(image).strip()
        items = ast.literal_eval(response)
        return items
    
    
    def detect_food(self, image):
        print("Food Classes", self.FOOD_CLASSES)

        cropped_image = image.copy()

        detections = self.grounding_dino_model.predict_with_classes(
            image=cropped_image,
            classes=self.FOOD_CLASSES,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
        )

        detections = self.remove_large_boxes(detections)
        

        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [
            f"{self.FOOD_CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _
            in detections]
        
        annotated_frame = box_annotator.annotate(scene=cropped_image.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            self.NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        if self.use_efficient_sam:
            result_masks = []
            for box in detections.xyxy:
                mask = efficient_sam_box_prompt_segment(image, box, self.efficientsam)
                result_masks.append(mask)
            detections.mask = np.array(result_masks)
        
        else:
            # Prompting SAM with detected boxes
            def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
                sam_predictor.set_image(image)
                result_masks = []
                # find the average center of all detected boxes
                center_xs = []
                center_ys = []
                mask_centers = []
                mask_scores = []

                for box in xyxy:
                    masks, scores, logits = sam_predictor.predict(
                        box=box,
                        multimask_output=True
                    )

                    x_min, y_min, x_max, y_max = box
                    centroid_x = (x_min + x_max) / 2
                    centroid_y = (y_min + y_max) / 2

                    # this is ugly
                    mask_centers.append((centroid_x, centroid_y))
                    center_xs.append(centroid_x)
                    center_ys.append(centroid_y)

                    index = np.argmax(scores)
                    result_masks.append(masks[index])
                    mask_scores.append(scores[index])

                # find the average center of all detected boxes
                center_x = int(np.mean(center_xs))
                center_y = int(np.mean(center_ys))
                total_center = (center_x, center_y)
                

                # Find the nearest box to the average center
                distances = [np.linalg.norm(np.array(total_center) - np.array(mask_center)) for mask_center in mask_centers]
                nearest_box_idx = np.argsort(distances)


                # Organize result_masks  and mask_scores according to nearest_box_idx
                result_masks = np.array(result_masks)[nearest_box_idx]
                mask_scores = np.array(mask_scores)[nearest_box_idx]

                # get the scores and masks of the three closest boxes to the center
                closest_masks = result_masks[:2]
                closest_mask_scores = mask_scores[:2]
                
                # organize the first three masks from highest to lowest score
                score_idx = np.argsort(closest_mask_scores)[::-1]

                closest_masks = closest_masks[score_idx]

                result_masks[:2] = closest_masks


                return result_masks

            # convert detections to masks
            detections.mask = segment(
                sam_predictor=self.sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )

        # annotate image with detections
        box_annotator = sv.BoundingBoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels = [
            f"{self.FOOD_CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _
            in detections]

        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        individual_masks = []
        refined_labels = []

        max_prob = 0
        max_prob_idx = None
        to_remove_idxs = []

        if len(to_remove_idxs) > 1:
            to_remove_idxs.remove(max_prob_idx)
            idxs = [i for i in range(len(detections)) if not i in to_remove_idxs]
        else:
            idxs = list(range(len(detections)))
        
        for i in range(len(detections)):
            mask_annotator = sv.MaskAnnotator(color=sv.Color.WHITE)
            H,W,C = image.shape
            mask = np.zeros_like(image).astype(np.uint8)
            d = sv.Detections(xyxy=detections.xyxy[i].reshape(1,4), \
                              mask=detections.mask[i].reshape((1,H,W)), \
                              class_id = np.array(detections.class_id[i]).reshape((1,)))
            mask = mask_annotator.annotate(scene=mask, detections=d)
            binary_mask = np.zeros((H,W)).astype(np.uint8)
            ys,xs,_ = np.where(mask > (0,0,0))
            binary_mask[ys,xs] = 255
            if i in idxs:
                individual_masks.append(binary_mask)
                refined_labels.append(labels[i])

        labels = refined_labels

        plate_mask = detect_plate(image)

        refined_masks = []
        portion_weights = []
        for i in range(len(individual_masks)):
            mask = individual_masks[i]
            label = labels[i]

            clean_mask = cleanup_mask(mask)

            refined_masks.append(clean_mask)

            food_enclosing_mask = clean_mask.copy()

            MIN_WEIGHT = 0.008
            portion_weights.append(max(1, mask_weight(food_enclosing_mask)/MIN_WEIGHT))


        return annotated_image, detections, refined_masks, portion_weights, labels
    
    def remove_large_boxes(self, detections):
        # Calculate average width and height of detected boxes
        total_width = 0
        total_height = 0
        num_boxes = len(detections.xyxy)

        for box in detections.xyxy:
            x_min, y_min, x_max, y_max = box        
            width = x_max - x_min
            height = y_max - y_min
            total_width += width
            total_height += height

        if num_boxes > 0:
            avg_width = total_width / num_boxes
            avg_height = total_height / num_boxes
        else:
            avg_width = 0
            avg_height = 0
        
        print("Average Width: ", avg_width, "Average Height: ", avg_height)

        # Filter out boxes that are bigger than the average box size
        filter_indices = [
            i for i, box in enumerate(detections.xyxy)
            if (box[2] - box[0]) <=  0.5 * avg_width and (box[3] - box[1]) <= avg_height
        ]

        detections.xyxy = detections.xyxy[filter_indices]
        detections.confidence = detections.confidence[filter_indices]
        detections.class_id = detections.class_id[filter_indices]

        return detections
    

    def clean_labels(self, labels):
        clean_labels = []
        instance_count = {}
        for label in labels:
            label = label[:-4].strip()
            if label != 'plate':
                clean_labels.append(label)
            if label in instance_count:
                instance_count[label] += 1
            else:
                instance_count[label] = 1
        return clean_labels, instance_count
    

    def categorize_items(self, labels, sim=True):
        categories = []

        if sim:
            for label in labels:
                if label in ['carrot']:
                    categories.append('carrot')
                elif label in ['pretzel']:
                    categories.append('pretzel')
                elif label in ['celery']:
                    categories.append('celery')
                

        return categories
    
    def get_autonomous_action(self, annotated_image, image, masks, categories, labels, portions, continue_food_label = None):
        vis = image.copy()

        if continue_food_label is not None:
            food_to_consider = [i for i in range(len(labels)) if labels[i] == continue_food_label]
            # TODO: Implement this
        else:
            food_to_consider = range(len(categories))

        print('Food to consider: ', food_to_consider)

        for idx in food_to_consider:
            if categories[idx] == 'carrot' or categories[idx] == 'pretzel' or categories[idx] == 'celery':
                self.get_grasp_action(image, masks, categories)


    
    def get_grasp_action(self, image, masks, categories):
        # changing to 1 so it closes all the way
        # self.robot_controller.set_gripper(0.65)

        solid_mask = None
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()

        for i, (category, mask) in enumerate(zip(categories, masks)):
            if category == 'carrot' or category == 'pretzel' or category == 'celery':
               for item_mask in mask:
                    centroid = detect_centroid(item_mask)
                    #clicks = self.pixel_selector.run(camera_color_data)
                    #centroid = clicks[0]

                    # get the coordinates of midpoints and points for width calculation
                    p1,p2,width_p1,width_p2,box = raf_utils.get_box_points(item_mask)
                    yaw_angle = raf_utils.pretzel_angle_between_pixels(p1,p2)

                    

                   
                   


                    lower_center = detect_lower_center(item_mask)
                    grasp_point = get_grasp_points(centroid, lower_center)
                   # vis2 = self.draw_points(camera_color_data, centroid, lower_center, grasp_point)



                    if not self.AUTONOMY:
                        vis2 = self.draw_points(camera_color_data, centroid, p1, p2, box)

                        cv2.imshow('vis2', vis2)
                        cv2.waitKey(0)
                        k = input("Is the grasp point correct? (y/n): ")
                        while k not in ['y', 'n']:
                            k = ('Is the grasp point correct? (y/n): ')
                            if k == 'e':
                                sys.exit(1)
                        while k == 'n':
                            sys.exit(1)
                        cv2.destroyAllWindows()
                    
                   
                    
                    #food_angle = raf_utils.angle_between_pixels(centroid, lower_center, camera_color_data.shape[1], camera_color_data.shape[0])
                    


                    #yaw_angle = raf_utils.pretzel_angle_between_pixels(centroid, lower_center)
                    #yaw_angle = raf_utils.pretzel_angle_between_pixels(p1,p2)

                    print("DETECTED ANGLE: ", yaw_angle)

                    
                    angle_of_rotation = (180-yaw_angle) - 30
                    rot = self.get_rotation_matrix(radians(angle_of_rotation))
                    
                    validity, center_point = raf_utils.pixel2World(camera_info_data, centroid[0], centroid[1], camera_depth_data)

                    # for the width calculation
                    validity, width_point1 = raf_utils.pixel2World(camera_info_data, width_p1[0], width_p1[1], camera_depth_data)
                    validity, width_point2 = raf_utils.pixel2World(camera_info_data, width_p2[0], width_p2[1], camera_depth_data)

                    
                    # width of object in cm
                    width = np.linalg.norm(width_point1 - width_point2)

                    # variables for the finger and pads
                    finger_offset = 0.6 # cm, how much the finger moves inwards from gripper
                    pad_offset = 0.35 # cm, thickness of a pad on fingertip
                    insurance = 0.97 # extra space for the object (100 is closed, 0 is open)
                    close = 1.175 # how much the gripper closes

                    # function transforming width to a gripper value 
                    grip_val = -7*((width*100)+(2*(finger_offset+pad_offset))) + 100
                    grip_val = grip_val*insurance

                    # make sure it doesnt exeed kinova limits
                    if grip_val > 100 - 2*(finger_offset+pad_offset):
                        grip_val = 100 - 2*(finger_offset+pad_offset)
                    elif grip_val < 0:
                        grip_val = 0

                    grip_val = round(grip_val)/100

                    print("width of food item: ", width)
                    print("Grip Value: ", grip_val)
                    

                    if not validity:
                        print("Invalid centroid")
                        continue
                    
                    # we're just saying its 7cm closer than the depth value to account for the gripper mod
                    center_point[2] -= 0.064

                    

                    food_transform = np.eye(4)
                    food_transform[:3,3] = center_point.reshape(1,3)
                    food_transform[:3,:3] = rot
                    logging_transform= food_transform.copy()

                    # gets the position of the camera link relative to the base link via transformation matrix
                    # multiplies with the camera to food TF
                    # result: food coordinates and orientation relative to the base link
                    food_base = self.tf_utils.getTransformationFromTF("base_link", "camera_link") @ food_transform
                    
                    

                    pose = self.tf_utils.get_pose_msg_from_transform(food_base)
                    
                    pose.position.y += self.CAMERA_OFFSET #.035
                    pose.position.z -= self.Z_OFFSET #0.0095
                    pose.position.x += 0.005 #tilt of realsense to kinova moves desire pos down to 0.005
                    #pose.position.z += self.GRIPPER_OFFSET # 0.0


                    euler_angles = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
                    
                    print(f"Food Pose: x:{pose.position.x}, y:{pose.position.y}, z:{pose.position.z} r:{degrees(euler_angles[0])}, p:{degrees(euler_angles[1])}, y:{degrees(euler_angles[2])}")
                    roll = (euler_angles[0])
                    pitch = euler_angles[1]
                    yaw = euler_angles[2] - 90
                    print("Food Angle: ", degrees(yaw)) 

                    q = quaternion_from_euler(roll, pitch, yaw)

                    pose.orientation.x = q[0]
                    pose.orientation.y = q[1]
                    pose.orientation.z = q[2]
                    pose.orientation.w = q[3]

                    
                    self.robot_controller.set_gripper(grip_val)
                    rospy.sleep(0.8)
                    
                    move_success =  self.robot_controller.move_to_pose(pose)

                    
                    if not self.AUTONOMY:
                        if move_success:
                            k = input("is the robot in the correct position? (y/n): ")
                            while k not in ['y', 'n']:
                                k = ('Is the robot in the correct position? (y/n): ')
                                if k == 'e':
                                    sys.exit(1)
                            while k == 'n':
                                sys.exit(1)
                                break
                        
                        grasp_success = self.robot_controller.set_gripper(grip_val*close)

                        k = input("Did the robot grasp the object? (y/n): ")
                        while k not in ['y', 'n']:
                            k = ('Did the robot grasp the object? (y/n): ')
                            if k == 'e':
                                sys.exit(1)
                        if k == 'n':
                            pose.position.z += 0.1
                            self.robot_controller.move_to_pose(pose)
                            self.clear_plate()
                            
                    else:
                        grasp_success = self.robot_controller.set_gripper(grip_val*close)

                    pose.position.z += 0.15
                    self.robot_controller.move_to_pose(pose)
                    time.sleep(3)
                    isGrasped = self.isObjectGrasped()
                    print("Is object grasped: ", isGrasped)
                    if not isGrasped:
                        self.robot_controller.reset()
                        time.sleep(3)
                        self.clear_plate()
                        break
                    self.robot_controller.move_to_feed_pose()

                    if not self.AUTONOMY:
                        input("Is user ready? (y/n): ")
                        while k not in ['y', 'n']:
                            k = ('Is the robot in the correct position? (y/n): ')
                            if k == 'e':
                                sys.exit(1)
                        while k == 'n':
                            sys.exit(1)
                            break
                        self.robot_controller.set_gripper(0.6)

                    time.sleep(2)

                    if self.gpt4v_client.PREFERENCE == "alternate":
                        if self.gpt4v_client.previous_bite == 'carrot':
                            self.gpt4v_client.update_history('celery')
                        else:
                            self.gpt4v_client.update_history('carrot')
                        self.robot_controller.reset()

                    if not self.AUTONOMY:
                        input("Continue feeding? (y/n): ")
                        while k not in ['y', 'n']:
                            k = ('Continue feeding? (y/n): ')
                            if k == 'e' or k == 'n':
                                sys.exit(1)
                                break
                        while k == 'n':
                            sys.exit(1)
                            break
                        self.robot_controller.reset()
                        self.clear_plate()
                        break
                    else:
                        time.sleep(3)
                        self.robot_controller.reset()
                        self.clear_plate()
                        break


    def isObjectGrasped(self):
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        x = 770
        y = 595
        validity, points = raf_utils.pixel2World(camera_info_data, x, y, camera_depth_data)
        if points is not None:
            points = list(points)
            if points[2] > 0.200 and points[2] < 0.300:
                return True
            else:
                return False 
        else: 
            return False                 

    def draw_points(self, image, center, lower, mid, box=None):
        if box is not None:
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        cv2.circle(image, center, 5, (0,255,0), -1)
        cv2.circle(image, lower, 5, (0,0,255), -1)
        cv2.circle(image, mid, 5, (0,0,255), -1)
        return image  


    def get_rotation_matrix(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])       

    
        





    

