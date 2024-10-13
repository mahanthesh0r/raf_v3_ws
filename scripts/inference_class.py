#!/usr/bin/env python3
import os
import base64
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

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
            }
        self.prompt_dir = prompt_dir

        
        with open("%s/identification.txt"%self.prompt_dir, 'r') as f:
            self.prompt_text = f.read()

    def encode_image(self, openCV_image):
        retval, buffer = cv2.imencode('.jpg', openCV_image)
        return base64.b64encode(buffer).decode('utf-8')
        
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

        self.Z_OFFSET = 0.0075
        self.CAMERA_OFFSET = 0.04
        
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.api_key = os.environ['OPENAI_API_KEY']
        self.gpt4v_client = GPT4VFoodIdentification(self.api_key, '/home/labuser/raf_v3_ws/src/raf_v3/scripts/prompts')

        self.FOOD_CLASSES = ["pretzel"]
        self.BOX_THRESHOLD = 0.3
        self.TEXT_THRESHOLD = 0.3
        self.NMS_THRESHOLD = 0.4

        self.camera = RealSenseROS()
        self.tf_utils = raf_utils.TFUtils()
        #self.skill_library = SkillLibrary()
        self.robot_controller = KinovaRobotController()

        
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

    def recognize_items(self, image):
        response = self.gpt4v_client.prompt(image).strip()
        items = ast.literal_eval(response)
        return items
    
    def detect_items(self, image):
        print("Food Classes", self.FOOD_CLASSES)

        cropped_image = image.copy()

        detections = self.grounding_dino_model.predict_with_classes(
            image=cropped_image,
            classes=self.FOOD_CLASSES,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
        )

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
                for box in xyxy:
                    masks, scores, logits = sam_predictor.predict(
                        box=box,
                        multimask_output=True
                    )
                    index = np.argmax(scores)
                    result_masks.append(masks[index])
                return np.array(result_masks)


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
    

    def clean_labels(self, labels):
        clean_labels = []
        instance_count = {}
        for label in labels:
            label = label[:-4].strip()
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
                if label in ['carrot', 'celery', 'pretzel']:
                    categories.append('snack')
                else:
                    categories.append('other')

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
            if categories[idx] == 'snack' or categories[idx] == 'other':
                self.get_grasp_action(image, masks, categories)

              
                


    
    def get_grasp_action(self, image, masks, categories):

        solid_mask = None
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()

        for i, (category, mask) in enumerate(zip(categories, masks)):
            if category == 'snack' or category == 'other':
               for item_mask in mask:
                    centroid = detect_centroid(item_mask)
                    lower_center = detect_lower_center(item_mask)
                    grasp_point = get_grasp_points(centroid, lower_center)
                    vis2 = self.draw_points(camera_color_data, centroid, lower_center, grasp_point)

                    cv2.imshow('vis2', vis2)
                    cv2.waitKey(0)
                    k = input("Is the grasp point correct? (y/n): ")
                    while k not in ['y', 'n']:
                        k = ('Is the grasp point correct? (y/n): ')
                        if k == 'e':
                            exit(1)
                        cv2.destroyAllWindows()
                    
                    self.robot_controller.set_gripper(0.80)
                    
                    food_angle = raf_utils.angle_between_pixels(centroid, lower_center, camera_color_data.shape[1], camera_color_data.shape[0])
                    yaw_angle = raf_utils.pretzel_angle_between_pixels(centroid, lower_center)
                    angle_of_rotation = (180-yaw_angle) - 30
                    rot = self.get_rotation_matrix(radians(angle_of_rotation))
                    
                    validity, center_point = raf_utils.pixel2World(camera_info_data, grasp_point[0], grasp_point[1], camera_depth_data)

                    if not validity:
                        print("Invalid centroid")
                        continue

                    food_transform = np.eye(4)
                    food_transform[:3,3] = center_point.reshape(1,3)
                    food_transform[:3,:3] = rot
                    logging_transform= food_transform.copy()
                    food_base = self.tf_utils.getTransformationFromTF("base_link", "camera_link") @ food_transform
                    


                    pose = self.tf_utils.get_pose_msg_from_transform(food_base)
                    
                    pose.position.y += self.CAMERA_OFFSET
                    pose.position.z -= self.Z_OFFSET


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

                    

                    
                    move_success =  self.robot_controller.move_to_pose(pose)

                    if move_success:
                        k = input("is the robot in the correct position? (y/n): ")
                        while k not in ['y', 'n']:
                            k = ('Is the robot in the correct position? (y/n): ')
                            if k == 'e':
                                exit(1)
                        
                        grasp_success = self.robot_controller.set_gripper(0.90)

                        if grasp_success:
                            pose.position.z += 0.1
                            move_success2 = self.robot_controller.move_to_pose(pose)
                            if move_success2:
                               feed_pose = self.robot_controller.move_to_feed_pose()
                               if feed_pose:
                                   self.robot_controller.set_gripper(0.6)
                                   self.robot_controller.reset()
                            
                    

                    

                    # success = self.skill_library.grasp_object('pose', pose, vel=0.5, accel=0.5, attempts=10, time=20, constraints=None)
                    # if success:
                    #     print("Successfully moved to the food item")
                        
                    # else:
                    #     print("Failed to move to the food item")

                    break
               


    def draw_points(self, image, center, lower, mid):
        cv2.circle(image, center, 5, (0,255,0), -1)
        cv2.circle(image, lower, 5, (0,0,255), -1)
        cv2.circle(image, mid, 5, (255,0,0), -1)
        return image  


    def get_rotation_matrix(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])       

    
        





    

