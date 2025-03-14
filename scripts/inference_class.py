#!/usr/bin/env python3
import os
import base64
import time
import copy
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
from vision_utils import efficient_sam_box_prompt_segment, detect_plate, cleanup_mask, mask_weight, detect_centroid, detect_lower_center, get_grasp_points, mask_width_points
import raf_utils
from rs_ros import RealSenseROS
from scipy.spatial.transform import Rotation
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_slerp
from math import sqrt, inf, degrees, radians
from robot_controller.robot_controller import KinovaRobotController
from pixel_selector import PixelSelector
import rospy
import re
from std_msgs.msg import String
from datetime import datetime, timedelta, date
from time import sleep
# from groundingdino.util.inference import Model
# from segment_anything import sam_model_registry, SamPredictor

# Grounded-SAM 2
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from skill_library import SkillLibrary


PATH_TO_GROUNDED_SAM = '/home/labuser/raf_v3_ws/Grounded-Segment-Anything'
PATH_TO_DEPTH_ANYTHING = '/home/labuser/raf_v3_ws/Depth-Anything'
USE_EFFICIENT_SAM = False

PATH_TO_GROUNDED_SAM2 = '/home/labuser/raf_v3_ws/Grounded-SAM-2'
API_TOKEN = "2b619f1c4b7434549812bae4690e52d8" # mahanthesh token
# API_TOKEN = "18a990ad81b63065ccd2aefb1c0bab77" # jake token
TEXT_PROMPT = "pasta ."
IMG_PATH = PATH_TO_GROUNDED_SAM2 + "/notebooks/images/cars.jpg"
SAM2_CHECKPOINT = PATH_TO_GROUNDED_SAM2 + "/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
BOX_THRESHOLD = 0.25
WITH_SLICE_INFERENCE = False
SLICE_WH = (480, 480)
OVERLAP_RATIO = (0.2, 0.2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_dinox_demo")
DUMP_JSON_RESULTS = True

sys.path.append(PATH_TO_GROUNDED_SAM2)

# from depth_anything.dpt import DepthAnything
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

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
        

    def encode_image(self, openCV_image):
        retval, buffer = cv2.imencode('.jpg', openCV_image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_history_food(self):
        with open("%s/history.txt"%self.history_file_path, 'r') as f:
            history = f.read()
            previous_bite = ast.literal_eval(history)
            print("Previous Bite: ", previous_bite[-1])
            return previous_bite[-1]
        
    def update_history(self, food):
        with open("%s/history.txt"%self.history_file_path, 'r') as f:
            history = f.read()
            previous_bite = ast.literal_eval(history)
            previous_bite.append(food)
            with open("%s/history.txt"%self.history_file_path, 'w') as f:
                f.write(str(previous_bite))
     
        
    def prompt(self, image):
        if self.PREFERENCE == "alternate":
            with open("%s/alternating_prompt.txt"%self.prompt_dir, 'r') as f:
                self.prompt_text = f.read()
                self.previous_bite = self.get_history_food()
                self.prompt_text = self.prompt_text.replace("{variable}", self.previous_bite)
        elif self.PREFERENCE == "carrots_first":
             with open("%s/carrots_first_prompt.txt"%self.prompt_dir, 'r') as f:
                 self.prompt_text = f.read()
        else:
            with open("%s/identification.txt"%self.prompt_dir, 'r') as f:
                self.prompt_text = f.read()

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
        self.command_stack = []
        rospy.Subscriber('speech_commands', String, self.command_callback)
        with torch.no_grad():
            torch.cuda.empty_cache()

        self.Z_OFFSET = 0.01
        self.GRIPPER_OFFSET = 0.00 #0.07
        self.CAMERA_OFFSET = 0.04
        self.AUTONOMY = False
        self.isPlate = True
        self.image_path = None
        self.logging_file_path = "/home/labuser/raf_v3_ws/src/raf_v3/scripts"
        self.isRetryAttempt = False
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.api_key = os.environ['OPENAI_API_KEY']
        self.gpt4v_client = GPT4VFoodIdentification(self.api_key, '/home/labuser/raf_v3_ws/src/raf_v3/scripts/prompts')
        self.FOOD_CLASSES = []
        self.BOX_THRESHOLD = 0.3
        self.TEXT_THRESHOLD = 0.3
        self.NMS_THRESHOLD = 0.4
        self.camera = RealSenseROS()
        self.tf_utils = raf_utils.TFUtils()
        self.robot_controller = KinovaRobotController()
        self.pixel_selector = PixelSelector()
        self.skill_library = SkillLibrary()
        # Grounding DINO stuff
        self.GROUNDING_DINO_CONFIG_PATH = PATH_TO_GROUNDED_SAM + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.GROUNDING_DINO_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/groundingdino_swint_ogc.pth"
         # Building GroundingDINO inference model
        #self.grounding_dino_model = Model(model_config_path=self.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH)
        self.use_efficient_sam = USE_EFFICIENT_SAM

        # Grounded-SAM 2
        self.API_TOKEN = "2b619f1c4b7434549812bae4690e52d8"
        self.text_prompt = "gummy bears . pretzel bites ."
        self.sam2_checkpoint = PATH_TO_GROUNDED_SAM2 + "/checkpoints/sam2.1_hiera_large.pt"
        self.sam2_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.BOX_THRESHOLD = 0.1
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # if self.use_efficient_sam:
        #     self.EFFICIENT_SAM_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/efficientsam_s_gpu.jit"
        #     self.efficientsam = torch.jit.load(self.EFFICIENT_SAM_CHECKPOINT_PATH)   

        # else:
        #     # Segment-Anything checkpoint
        #     SAM_ENCODER_VERSION = "vit_h"
        #     SAM_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/sam_vit_h_4b8939.pth"

        #     # Building SAM Model and SAM Predictor
        #     sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        #     sam.to(device=self.DEVICE)
        #     self.sam_predictor = SamPredictor(sam)

        # self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(self.DEVICE).eval()
        # self.depth_anything_transform = Compose([
        #     Resize(
        #         width=518,
        #         height=518,
        #         resize_target=False,
        #         keep_aspect_ratio=True,
        #         ensure_multiple_of=14,
        #         resize_method='lower_bound',
        #         image_interpolation_method=cv2.INTER_CUBIC,
        #     ),
        #     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     PrepareForNet(),
        # ])

    def command_callback(self, msg):
        """
        Callback function to handle incoming commands.
        msg: The message received from the speech_commands topic.
        """
        command = msg.data
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

    def logging(self, success=0, retries=0):
        log_file_path = "%s/logging.txt" % self.logging_file_path
        previous_success = 0
        previous_retries = 0
        # Read the current values from the log file
        try:
            with open(log_file_path, 'r') as f:
                content = f.read()
                match = re.search(r"Success: (\d+), retries: (\d+)", content)
                if match:
                    previous_success = int(match.group(1))
                    previous_retries = int(match.group(2))
        except FileNotFoundError:
            # If the file does not exist, start with initial values
            pass

        # Increment the values
        new_success = previous_success + success
        new_retries = previous_retries + retries

        # Write the updated values to the log file
        with open(log_file_path, 'w') as f:
            f.write("Success: %d, retries: %d" % (new_success, new_retries))

    def clear_plate2(self):
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        self.detect_food_GS2(camera_color_data)

    # def clear_plate(self, cup=False):
    #     self.listen_for_commands(3)
    #     self.FOOD_CLASSES = []
    #     camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
    #     if not cup:
    #         self.robot_controller.reset()
    #         items = self.recognize_items(camera_color_data)
    #         #items = ['gummy bears']
    #         items = raf_utils.randomize_selection(items)

    #         if items is not None:
    #             print("Items: ", items)
    #             if items != [] and 'watermelon' in items:
    #                 items = ['red cube' if item == 'watermelon' else item for item in items]
    #             if items != [] and 'chocolate' in items:
    #                 items = ['brown cube' if item == 'chocolate' else item for item in items]
            
    #         # added by jake
    #         else:
    #             print("No food items detected!")
    #             sys.exit(1)

            
    #         #items = ['brown cube','grapes']
    #     else:
    #         self.robot_controller.move_to_cup_joint()
    #         rospy.sleep(5)
    #         #items = self.recognize_items(camera_color_data)
    #         items = ['cup']

    #     print("Items: ", items)
    #     self.FOOD_CLASSES = items

    #     if camera_color_data is None:
    #         print("No camera data")
    #         return
        
    #     annotated_image, detections, item_masks, item_portions, item_labels = self.detect_food(camera_color_data, isCup=cup)
       
    #     if not self.AUTONOMY:
    #         vis = camera_color_data.copy()
    #         cv2.imshow('vis', annotated_image)
    #         cv2.waitKey(0)
    #         if not raf_utils.validate_with_user("Are the detected items correct? "):
    #             sys.exit(1)
    #         cv2.destroyAllWindows()

    #     clean_item_labels, _ = self.clean_labels(item_labels)
    #     print("----- Clean Item Labels:", clean_item_labels)

    #     categories = self.categorize_items(clean_item_labels)
    #     print("--------------------")
    #     print("Labels:", item_labels)
    #     print("Categories:", categories)
    #     print("Portions:", item_portions)
    #     print("--------------------")



    #     category_list, labels_list, per_food_masks, per_food_portions = raf_utils.organize_food_data(categories, clean_item_labels, item_masks, item_portions)

    #     print("Category List:", category_list)
    #     print("Labels List:", labels_list)
    #     print("Per Food Masks Len:", [len(x) for x in per_food_masks])
    #     print("Per Food Portions:", per_food_portions)

    #     self.get_autonomous_action(annotated_image, camera_color_data, per_food_masks, category_list, labels_list, per_food_portions)


    def listen_for_commands(self, duration):
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration)
        while datetime.now() < end_time:
            print("Listening...")
            print("Voice Command: ", self.get_command())
            sleep(1)
            if self.get_command() == 'stop':
                self.clear_stack()
                sys.exit(1)
                return
            elif self.get_command() == 'feed':
                self.clear_stack()
            elif self.get_command() == 'drink':
                self.clear_stack()
                self.clear_plate(cup=True)


    def recognize_items(self, image):
        response = self.gpt4v_client.prompt(image).strip()
        items = ast.literal_eval(response)
        return items
    
    def detect_food_GS2(self, image, isCup=False):
        if image is None:
            print("No camera data captured")
            return
        config = Config(self.API_TOKEN)
        client = Client(config)
        classes = [x.strip().lower() for x in self.text_prompt.split('.') if x]
        class_name_to_id = {name: id for id, name in enumerate(classes)}
        # cv2.imshow("Captured Image", image)
        # cv2.waitKey(0)
        self.image_path = raf_utils.image_from_camera(image)
        print("Image Path: ", self.image_path)
        
        image_url = client.upload_file(self.image_path)
        task = DinoxTask(
            image_url=image_url,
            prompts=[TextPrompt(text=self.text_prompt)],
            bbox_threshold=BOX_THRESHOLD,
            targets=[DetectionTarget.BBox],
        )
        client.run_task(task)
        result = task.result
        objects = result.objects
        input_boxes = []
        confidences = []
        class_names = []
        class_ids = []

        for idx, obj in enumerate(objects):
            input_boxes.append(obj.bbox)
            confidences.append(obj.score)
            cls_name = obj.category.lower().strip()
            class_names.append(cls_name)
            class_ids.append(class_name_to_id[cls_name])
        
        input_boxes = np.array(input_boxes)
        class_ids = np.array(class_ids)
        
        torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=DEVICE)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        image = Image.open(self.image_path)
        sam2_predictor.set_image(np.array(image.convert("RGB")))
        
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        print("Here 2")
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        labels = [
            f"{class_name}: {confidence:.2f}"
            for class_name, confidence in zip(class_names, confidences)
        ]
        img = cv2.imread(self.image_path)
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
        )
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imshow('annotated_frame', annotated_frame)
        cv2.waitKey(0)
        
        

    
    
    def detect_food(self, image, isCup=False):
        print("Food Classes", self.FOOD_CLASSES)
        cropped_image = image.copy()
        category = raf_utils.get_category_from_label(self.FOOD_CLASSES)
        print("Category: ", category)

        if category == 'plate_snack':
            self.BOX_THRESHOLD = 0.3
            self.TEXT_THRESHOLD = 0.3
            self.NMS_THRESHOLD = 0.4
        elif category == 'bowl_snack':
            # if cropping, all threshholds should be higher
            self.BOX_THRESHOLD = 0.036
            self.TEXT_THRESHOLD = 0.028
            self.NMS_THRESHOLD = 0.4 
        elif category == 'meal': #or category == 'special_meal':
            self.BOX_THRESHOLD = 0.5
            self.TEXT_THRESHOLD = 0.506
            self.NMS_THRESHOLD = 0.4
        elif category == 'drink':
            self.BOX_THRESHOLD = 0.3
            self.TEXT_THRESHOLD = 0.3
            self.NMS_THRESHOLD = 0.4
        elif category == 'test':
            self.BOX_THRESHOLD = 0.3
            self.TEXT_THRESHOLD = 0.3
            self.NMS_THRESHOLD = 0.4
        elif category == 'special_meal':
            self.BOX_THRESHOLD = 0.3
            self.TEXT_THRESHOLD = 0.3
            self.NMS_THRESHOLD = 0.4
        elif category == 'pasta':
            self.BOX_THRESHOLD = 0.01
            self.TEXT_THRESHOLD = 0.01
            self.NMS_THRESHOLD = 0.4 


        print("Thresholds: ", self.BOX_THRESHOLD, self.TEXT_THRESHOLD, self.NMS_THRESHOLD)

        detections = self.grounding_dino_model.predict_with_classes(
            image=cropped_image,
            classes=self.FOOD_CLASSES,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
        )

        # if not self.isPlate and not isCup:
        #detections = self.remove_large_boxes(detections)
        
        # make sure plate isnt detected as food item
        # if not isCup and category != 'pasta':
        #     detections = self.remove_plate(detections)
        # elif category == 'pasta':
        #     detections = self.remove_plate(detections, food_container='bowl')
        

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
            if self.gpt4v_client.PREFERENCE == "custom" and not self.isPlate and not isCup:

                # Prompting SAM with detected boxes
                def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
                    sam_predictor.set_image(image)
                    result_masks = []

                    print("picking towards center")
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
            else:
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
    
    # ensure that the plate is not detected as an object
    def remove_plate(self, detections, food_container='plate'):

        # size of plate bounding box when in overlook position
        plate_area = 362964
        
        if food_container != 'plate':
            # heinens box size
            plate_area = 100000
        
        print("Plate Area: ", plate_area)
        # tolerance of how close the area can be to the plate area (percentage)
        large_box_tolerance = 0.65

        # Filter out boxes that are plate sized
        filter_indices = [
            i for i, box in enumerate(detections.xyxy)
            if abs(plate_area-abs((box[2] - box[0]) * (box[3] - box[1]))) >= plate_area * large_box_tolerance
        ]

        print("Plate Filter Indices: ", filter_indices)
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
                if label in ['carrot', 'celery', 'small rod pretzel']:
                    categories.append('plate_snack')
                elif label in ['almonds', 'pretzel bites', 'green grapes', 'french fries', 'fruits','gummy bears', 'gummy worms', 'brown cube','red cube', 'pretzel rods','penne pasta','tomato','green vegetable', 'chicken nugget']:
                    categories.append('bowl_snack')
                elif label in ['dumplings', 'chicken tenders','egg rolls']:
                    categories.append('meal')
                elif label in ['cup', 'bottle']:
                    categories.append('drink')
                elif label in ['sushi', 'donut']:
                    categories.append('special_meal')
                elif label in ['penne pasta','tomato','green vegetable']:
                    categories.append('pasta') 
                
        return categories
    
    def get_autonomous_action(self, annotated_image, image, masks, categories, labels, portions, continue_food_label = None):
        vis = image.copy()


        if continue_food_label is not None:
            food_to_consider = [i for i in range(len(labels)) if labels[i] == continue_food_label]
            # TODO: Implement this
        else:
            food_to_consider = range(len(categories))

        print('Food to consider: ', food_to_consider)

        # Check voice command stack
        if self.get_command() == 'stop':
            self.clear_stack()
            sys.exit(1)
            return
        elif self.get_command() == 'drink':
            self.clear_stack()
            self.clear_plate(cup=True)
            return

        for idx in food_to_consider:
            if categories[idx] == 'plate_snack':
                self.grasp_plate_snack(image, masks, categories,close=1.12)
            elif categories[idx] == 'meal':
                self.grasp_plate_snack(image, masks, categories)
            elif categories[idx] == 'bowl_snack' :
                self.get_grasp_action(image, masks, categories)
            elif categories[idx] == 'drink':
               self.grasp_drink(image, masks, categories)
            elif categories[idx] == 'special_meal' or categories[idx] == 'pasta':
                self.get_grasp_action(image, masks, categories, close=1.27)

    

    def grasp_plate_snack(self, image, masks, categories, finger_offset=0.6, pad_offset=0.35, insurance=0.985, close=1.26):
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        for i, (category, mask) in enumerate(zip(categories, masks)):
            if category == 'plate_snack' or category == 'meal':
                for item_mask in mask:
                    grasp_point, centroid, yaw_angle, wp1, wp2, p1, p2 = self.calculate_grasp_point_width(item_mask, 'plate_snack')

                    if not self.AUTONOMY:
                        vis2 = self.draw_points(camera_color_data, grasp_point, p1, p2, wp1,wp2)
                        cv2.imshow('vis2', vis2)
                        cv2.waitKey(0)
                        if not raf_utils.validate_with_user("Is the grasp point correct? "):
                            sys.exit(1)
                        cv2.destroyAllWindows()    
                    pose, width_point1, width_point2 = self.get_object_position(camera_info_data, camera_depth_data, yaw_angle, grasp_point, wp1, wp2, 'plate_snack')
                    grip_val = self.skill_library.getGripperWidth(width_point1, width_point2, finger_offset, pad_offset, insurance, close)            
                    self.robot_controller.set_gripper(grip_val)
                    rospy.sleep(0.8)

                    move_success = self.robot_controller.move_to_pose(pose)
                    if not self.AUTONOMY:
                        if move_success:
                            if not raf_utils.validate_with_user("Is the robot in the correct position? "):
                                sys.exit(1)
                                break
                        #go down to grasp plate stack
                        pose.position.z -= 0.017
                        self.robot_controller.move_to_pose(pose)
                        rospy.sleep(1)
                        grasp_success = self.robot_controller.set_gripper(grip_val*close)
                        if not raf_utils.validate_with_user("Did the robot grasp the object? "):
                            pose.position.z += 0.1
                            self.robot_controller.move_to_pose(pose)
                            self.clear_plate()
                    else:
                        #go down to grasp plate stack
                        pose.position.z -= 0.017
                        self.robot_controller.move_to_pose(pose)
                        rospy.sleep(1)
                        grasp_success = self.robot_controller.set_gripper(grip_val*close)
                        
                    pose.position.z += 0.15
                    self.robot_controller.move_to_pose(pose)
                    time.sleep(2)
                    if not self.isObjectGrasped():
                        self.robot_controller.reset()
                        time.sleep(3)
                        self.isRetryAttempt = True
                        self.clear_plate()
                        return
                    self.robot_controller.move_to_multi_bite_transfer()

                    # Check for multi-bite
                    while self.checkObjectGrasped():
                        pass

                    # if not self.AUTONOMY:
                    #     input("Is user ready? (y/n): ")
                    #     while k not in ['y', 'n']:
                    #         k = ('Is the robot in the correct position? (y/n): ')
                    #         if k == 'e':
                    #             sys.exit(1)
                    #     while k == 'n':
                    #         sys.exit(1)
                    #         break
                    #     self.robot_controller.set_gripper(0.6)

                    time.sleep(2)

                    if self.gpt4v_client.PREFERENCE == "alternate":
                        if self.gpt4v_client.previous_bite == 'carrot':
                            self.gpt4v_client.update_history('celery')
                        else:
                            self.gpt4v_client.update_history('carrot')
                        self.robot_controller.reset()
                    
                    if not self.AUTONOMY:
                        if not raf_utils.validate_with_user("Continue feeding? (y/n): "):
                            sys.exit(1)
                            break
                        self.robot_controller.reset()
                        self.clear_plate()
                        break
                    else:
                        time.sleep(3)
                        self.robot_controller.reset()
                        if self.isRetryAttempt:
                            self.logging(retries=1)
                            self.isRetryAttempt = False
                        else:
                            self.logging(success=1)
                        self.clear_plate()
                        break

    # change here
    # changing close from 1.12 to 1.2  for pasta
    def get_grasp_action(self, image, masks, categories, finger_offset=0.6, pad_offset=0.35, insurance=0.985, close=1.185):
        solid_mask = None
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()

        for i, (category, mask) in enumerate(zip(categories, masks)):
            if category == 'bowl_snack' or category == 'special_meal':
               for item_mask in mask:                   
                    grasp_point, centroid, yaw_angle, wp1, wp2, p1, p2 = self.calculate_grasp_point_width(item_mask, 'bowl_snack')
                    if not self.AUTONOMY:
                        vis2 = self.draw_points(camera_color_data, centroid, p1, p2, wp1,wp2)

                        cv2.imshow('vis2', vis2)
                        cv2.waitKey(0)
                        if not raf_utils.validate_with_user("Is the grasp point correct? "):
                            sys.exit(1)
                        cv2.destroyAllWindows() 
                    pose, width_point1, width_point2 = self.get_object_position(camera_info_data, camera_depth_data, yaw_angle, centroid, wp1, wp2, 'bowl_snack')                  
                    grip_val = self.skill_library.getGripperWidth(width_point1, width_point2, finger_offset, pad_offset, insurance, close)
                    self.robot_controller.set_gripper(grip_val)
                    rospy.sleep(0.8)    
                    move_success =  self.robot_controller.move_to_pose(pose)     
                    if not self.AUTONOMY:
                        if move_success:
                            if not raf_utils.validate_with_user("Is the robot in the correct position? "):
                                sys.exit(1)
                                break
                        #added to change how far down the gripper goes for bowl snacks
                        pose.position.z -= 0.017
                        if category == 'special_meal':
                            pose.position.z -= 0.018
                        if category == 'pasta':
                            pose.position.z -= 0.02
                        self.robot_controller.move_to_pose(pose)
                        rospy.sleep(1)
                        grasp_success = self.robot_controller.set_gripper(grip_val*close)
                        if not raf_utils.validate_with_user("Did the robot grasp the object? "):
                            pose.position.z += 0.1
                            self.robot_controller.move_to_pose(pose)
                            self.clear_plate()        
                    else:
                        pose.position.z -= 0.017
                        self.robot_controller.move_to_pose(pose)
                        rospy.sleep(1)
                        grasp_success = self.robot_controller.set_gripper(grip_val*close)

                    pose.position.z += 0.15
                    self.robot_controller.move_to_pose(pose)
                    time.sleep(3)
                    if not self.isObjectGrasped():
                        self.robot_controller.reset()
                        time.sleep(3)
                        self.isRetryAttempt = True
                        self.clear_plate()
                        return
                    self.robot_controller.move_to_feed_pose()
                    # Check for multi-bite
                    while self.checkObjectGrasped():
                        pass

                    # if not self.AUTONOMY:
                    #     input("Is user ready? (y/n): ")
                    #     while k not in ['y', 'n']:
                    #         k = ('Is the robot in the correct position? (y/n): ')
                    #         if k == 'e':
                    #             sys.exit(1)
                    #     while k == 'n':
                    #         sys.exit(1)
                    #         break
                    #     self.robot_controller.set_gripper(0.6)

                    time.sleep(2)
                    
                    if self.gpt4v_client.PREFERENCE == "alternate":
                        if self.gpt4v_client.previous_bite == 'dumplings':
                            self.gpt4v_client.update_history('sushi')
                        else:
                            self.gpt4v_client.update_history('dumplings')
                        self.robot_controller.reset()

                    if not self.AUTONOMY:
                        if not raf_utils.validate_with_user("Continue feeding? (y/n): "):
                            sys.exit(1)
                            break
                        self.robot_controller.reset()
                        self.clear_plate()
                        break
                    else:
                        time.sleep(3)
                        #self.robot_controller.set_gripper(0.6)
                        self.robot_controller.reset()
                        if self.isRetryAttempt:
                            self.logging(retries=1)
                            self.isRetryAttempt = False
                        else:
                            self.logging(success=1)
                        self.clear_plate()
                        break

    def grasp_drink(self, image, masks, categories, finger_offset=0.6, pad_offset=0.35, insurance=0.975, close=1.175):
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        for i, (category, mask) in enumerate(zip(categories, masks)):
            if category == 'drink':
                for item_mask in mask:
                    grasp_point, centroid, yaw_angle, wp1, wp2, p1, p2 = self.calculate_grasp_point_width(item_mask, 'drink')
                    
                    if not self.AUTONOMY:
                        vis2 = self.draw_points(camera_color_data, centroid, p1, p2)
                        cv2.imshow('vis2', vis2)
                        cv2.waitKey(0)
                        if not raf_utils.validate_with_user("Is the grasp point correct? "):
                            sys.exit(1)
                        cv2.destroyAllWindows()
                    pose, width_point1, width_point2 = self.get_object_position(camera_info_data, camera_depth_data, yaw_angle, centroid, wp1, wp2, 'drink')
                    self.robot_controller.set_gripper(0.0)
                    rospy.sleep(0.8)
                    cup_position = copy.deepcopy(pose)
                    move_success = self.robot_controller.move_to_pose(pose)
                    if not self.AUTONOMY:
                        if move_success:
                           if not raf_utils.validate_with_user("Is the robot in the correct position? "):
                                sys.exit(1)
                                break
                        grasp_success = self.robot_controller.set_gripper(0.65)
                        if not raf_utils.validate_with_user("Did the robot grasp the object? "):
                            pose.position.z += 0.1
                            self.robot_controller.move_to_pose(pose)
                            self.clear_plate()
                    else:
                        grasp_success = self.robot_controller.set_gripper(0.65)
                        rospy.sleep(2)
                    pose.position.z += 0.15
                    self.robot_controller.move_to_pose(pose)
                    time.sleep(2)
                    self.robot_controller.move_to_sip_pose()
                    time.sleep(4)
                    self.robot_controller.move_to_cup_joint()
                    time.sleep(3)
                    self.robot_controller.move_to_pose(cup_position)
                    time.sleep(3)
                    self.robot_controller.set_gripper(0)
                    time.sleep(2)
                    cup_position.position.z += 0.25
                    self.robot_controller.move_to_pose(cup_position)
                    time.sleep(2)
                    self.robot_controller.reset()
                    self.clear_plate()
                    break

    def isObjectGrasped(self):
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        #For pretzels
        # x = 770
        # y = 595

        #grapes
        x = 763
        y = 571

        #chicken tenders
        # x = 722 
        # y = 588

        #For Carrots
        # x = 762
        # y = 576

        #For almonds
        # x = 766
        # y = 562
        validity, points = raf_utils.pixel2World(camera_info_data, x, y, camera_depth_data)
        if points is not None:
            points = list(points)
            print("Object Depth", points[2])
            # if points[2] > 0.200 and points[2] < 0.300:
            if points[2] < 0.300:
                return True
            else:
                return False 
        else: 
            return False    
    

    def checkObjectGrasped(self):
            consecutive_false_count = 0

            while True:
                grasped = self.isObjectGrasped()
                if grasped:
                    consecutive_false_count = 0
                    print("Object grasped: True")
                else:
                    consecutive_false_count += 1
                    print("Object grasped: False")

                if consecutive_false_count >= 2:
                    return False

                time.sleep(2)
                     

    def draw_points(self, image, center, lower, mid, wp1=None, wp2=None, box=None):
        if box is not None:
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        if wp1 is not None and wp2 is not None:
            cv2.line(image, wp1, wp2, (255,0,0), 2)
        cv2.circle(image, center, 5, (0,255,0), -1)
        cv2.circle(image, lower, 5, (0,0,255), -1)
        cv2.circle(image, mid, 5, (255,0,0), -1)
        return image  


    def get_rotation_matrix(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]]) 


    def calculate_grasp_point_width(self, item_mask, category):
        if category == 'plate_snack':
            centroid = detect_centroid(item_mask)
            # get the coordinates of midpoints and points for width calculation
            p1,p2,width_p1,width_p2,box = raf_utils.get_box_points(item_mask)

            # finds orientation of food
            yaw_angle = raf_utils.pretzel_angle_between_pixels(p1,p2)

            # find bottom of food
            lower_center = detect_lower_center(item_mask)
            
            # finds the point the gripper will go to 
            grasp_point = get_grasp_points(centroid, lower_center)

            # find the points on the box across from the grasp point
            gp1,gp2 = raf_utils.get_width_points(grasp_point, item_mask) # get the correct width for the gripper based on the mask
            

            # find the nearest point on the mask to the points
            wp1,wp2 = mask_width_points(gp1,gp2,item_mask)      

            return grasp_point, centroid, yaw_angle, wp1, wp2, p1, p2
        
        elif category == 'bowl_snack':
            centroid = detect_centroid(item_mask)
            p1,p2,width_p1,width_p2,box = raf_utils.get_box_points(item_mask)
            yaw_angle = raf_utils.pretzel_angle_between_pixels(p1,p2)
            lower_center = detect_lower_center(item_mask)
            grasp_point = get_grasp_points(centroid, lower_center)
            gp1,gp2 = raf_utils.get_width_points(centroid, item_mask) # get the correct width for the gripper based on the mask
            wp1,wp2 = mask_width_points(gp1,gp2,item_mask)

            return grasp_point, centroid, yaw_angle, wp1, wp2, p1, p2
        
        elif category == 'drink':
            centroid = detect_centroid(item_mask)
            p1,p2,width_p1,width_p2,box = raf_utils.get_cup_box_points(item_mask)
            far_right_point, far_left_point, centroid1, mask_with_points = raf_utils.get_cup_box_points_v2(item_mask)
            yaw_angle = raf_utils.pretzel_angle_between_pixels(p1,p2)
            lower_center = detect_lower_center(item_mask)
            grasp_point = get_grasp_points(centroid, lower_center) 

            return grasp_point, centroid, yaw_angle, far_right_point, far_left_point, p1, p2
        
    
    def get_object_position(self, camera_info_data, camera_depth_data, yaw_angle, grasp_point, wp1, wp2, category):
        print("food classes: ", self.FOOD_CLASSES)
        if category == 'drink':
            angle_of_rotation = 180
        else:
            angle_of_rotation = (180 - yaw_angle) - 30
        
        
        rot = self.get_rotation_matrix(radians(angle_of_rotation))
        validity, center_point = raf_utils.pixel2World(camera_info_data, grasp_point[0], grasp_point[1], camera_depth_data)
        # for the width calculation
        validity, width_point1 = raf_utils.pixel2World(camera_info_data, wp1[0], wp1[1], camera_depth_data)
        validity, width_point2 = raf_utils.pixel2World(camera_info_data, wp2[0], wp2[1], camera_depth_data)
        if not validity:
            print("Invalid centroid")
        if center_point is None and category is 'drink':
            print("No depth data! Retrying...")
            # recursive function! not good
            self.clear_plate(cup=True)
            return
        
        if category == 'drink':
            center_point[2] += 0.06
        else:
            center_point[2] -= 0.065
        food_transform = np.eye(4)
        food_transform[:3,3] = center_point.reshape(1,3)
        food_transform[:3,:3] = rot

        food_base = self.tf_utils.getTransformationFromTF("base_link", "camera_link") @ food_transform

        pose = self.tf_utils.get_pose_msg_from_transform(food_base)
        pose.position.y += self.CAMERA_OFFSET # Realsense camera offset
        pose.position.z -= self.Z_OFFSET # 0.01
        pose.position.x += 0.007# Tilt of realsense to kinova moves desire pos down to 0.005

        euler_angles = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        roll = euler_angles[0]
        pitch = euler_angles[1]
        yaw = euler_angles[2] - 90
        if category == 'drink':
            yaw = euler_angles[2]

        q = quaternion_from_euler(roll, pitch, yaw)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        return pose, width_point1, width_point2



    
        





    

