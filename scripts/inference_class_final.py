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
from vision_utils import detect_centroid, detect_lower_center, get_grasp_points, mask_width_points, cleanup_mask
import raf_utils
from rs_ros import RealSenseROS
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from math import radians
from robot_controller.robot_controller import KinovaRobotController
from pixel_selector import PixelSelector
import rospy
import re
from std_msgs.msg import String
from datetime import datetime, timedelta, date
from time import sleep
from skill_library import SkillLibrary
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

import yaml
from dotenv import load_dotenv
import logging
import ast
import colorlog
from raf_v3.msg import VisualServoData
from sort import Sort
import asyncio


# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/labuser/raf_v3_ws/src/raf_v3/scripts/logs/inference_class.log'),
        logging.StreamHandler()
    ])

class BiteAcquisitionInference:

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        load_dotenv()

        
        self.logger = logging.getLogger("inference_class_final")
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler('/home/labuser/raf_v3_ws/src/raf_v3/scripts/logs/inference_class.log')
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        color_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                }
            )
        console_handler.setFormatter(color_formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.Z_OFFSET = config['Z_OFFSET']
        self.GRIPPER_OFFSET = config['GRIPPER_OFFSET']
        self.CAMERA_OFFSET = config['CAMERA_OFFSET']
        self.PLATE_DEPTH = config['PLATE_DEPTH']
        self.AUTONOMY = config['AUTONOMY']
        self.image_path = None
        self.logging_file_path = config['logging_file_path']
        self.isRetryAttempt = config['isRetryAttempt']
        self.DEVICE = 'cuda' if config['DEVICE'] == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.api_key = os.environ['OPENAI_API_KEY']
        self.gpt4v_client = GPT4VFoodIdentification(self.api_key, '/home/labuser/raf_v3_ws/src/raf_v3/scripts/prompts')
        self.FOOD_CLASSES = []
        self.BOX_THRESHOLD = config['BOX_THRESHOLD']
        self.camera = RealSenseROS()
        self.tf_utils = raf_utils.TFUtils()
        self.robot_controller = KinovaRobotController()
        self.pixel_selector = PixelSelector()
        self.skill_library = SkillLibrary()
        self.dinox_api_key = os.getenv('DINOX_API_KEY')
        self.path_to_grounded_sam2 = config['PATH_TO_GROUNDED_SAM2']
        self.sam2_checkpoint = self.path_to_grounded_sam2 + config['SAM2_CHECKPOINT']
        self.sam2_model_config = config['SAM2_MODEL_CONFIG']
        self.command_stack = []
        self.bite_transfer_height = config['bite_transfer_height']
        

        # Publisher for the visual_servo_data topic
        self.visual_servo_pub = rospy.Publisher('visual_servo_data', VisualServoData, queue_size=10)

        rospy.Subscriber('speech_commands', String, self.command_callback)
        with torch.no_grad():
            torch.cuda.empty_cache()

    # Voice Recognition module
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


    def listen_for_commands(self, duration):
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration)
        while datetime.now() < end_time:
            self.logger.debug("Listening...") 
            self.logger.info(f"Voice Command:  {self.get_command()}")
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

    # Data Logging and Results
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


    def recognize_items(self, image):
        response = self.gpt4v_client.prompt(image).strip()
        items = ast.literal_eval(response)
        return items
    
    def food_selection(self, food_items):
        food_items = raf_utils.randomize_selection(food_items)
        self.logger.info(f"Food_Classes: {food_items}")
        detection_prompt = raf_utils.list_to_prompt_string(food_items)
        self.logger.info(f"Detection Prompt: {detection_prompt}")
        return detection_prompt
    
    def detect_food(self, image, prompt):
        config = Config(self.dinox_api_key)
        client = Client(config)
        classes = [x.strip().lower() for x in prompt.split('.') if x]
        class_name_to_id = {name: id for id, name in enumerate(classes)}
        self.image_path = raf_utils.image_from_camera(image)
        image_url = client.upload_file(self.image_path)
        task = DinoxTask(
            image_url=image_url,
            prompts=[TextPrompt(text=prompt)],
            bbox_threshold=self.BOX_THRESHOLD,
            targets=[DetectionTarget.BBox],
        )
        client.run_task(task)
        result = task.result
        objects = result.objects
        input_boxes = []
        confidences = []
        class_ids = []
        class_names = []

        for idx, obj in enumerate(objects):
            input_boxes.append(obj.bbox)
            confidences.append(obj.score)
            cls_name = obj.category.lower().strip()
            class_names.append(cls_name)
            class_ids.append(class_name_to_id[cls_name])

        input_boxes = np.array(input_boxes)
        class_ids = np.array(class_ids)
        if input_boxes.size == 0:
            return None, None, None, None

        torch.autocast(device_type=self.DEVICE, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.DEVICE)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        image = Image.open(self.image_path)
        sam2_predictor.set_image(np.array(image.convert("RGB")))

        masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
        )

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

        # index = np.argmax(scores)
        index = 0
        self.logger.debug(f"Scores: {scores}")
        self.logger.debug(f"Index: {index} : {scores[index]}")
        self.logger.debug(f"confidence: {confidences[index]}")

        H,W,C = img.shape
        mask = np.zeros_like(img).astype(np.uint8)
        self.logger.debug(f"Detec Box: {detections.xyxy[index]}")
        d = sv.Detections(xyxy=detections.xyxy[index].reshape(1,4), 
                         mask=detections.mask[index].reshape((1,H,W)), 
                         class_id=np.array(detections.class_id[index]).reshape((1,)))
        mask = mask_annotator.annotate(scene=mask, detections=d)
        binary_mask = np.zeros((H,W)).astype(np.uint8)
        ys,xs,_ = np.where(mask > (0,0,0))
        binary_mask[ys,xs] = 255
        label = labels[index]
        self.logger.debug(f"Label: {label}")
        refined_mask = cleanup_mask(binary_mask)

        return annotated_frame, d, refined_mask, label
    

    # def track_item(self, bbox):
    #     camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
    #     tracker = cv2.TrackerCSRT_create()
    #     vis = camera_color_data.copy()
    #     x, y, x2, y2 = map(int, bbox.xyxy[0])
    #     width = int(x2 - x)
    #     height = int(y2 - y)
    #     bbox_final = (x, y, width, height)
    #     self.logger.debug(f"bbox: {bbox_final}")
    #     tracker.init(vis, bbox_final)
    #     # Initialize the SAM2 for segmenting
    #     sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.DEVICE)
    #     sam2_predictor = SAM2ImagePredictor(sam2_model)
    #     while True:
    #         camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
    #         success, bbox = tracker.update(camera_color_data)
    #         if success:
    #             x, y, w, h = map(int, bbox)  # Unpack bbox correctly
    #             x2, y2 = x + w, y + h  # Convert (x, y, w, h) -> (x1, y1, x2, y2)


    #             # Draw tracking box
    #             cv2.rectangle(camera_color_data, (x, y), (x2, y2), (0, 0, 255), 2)

    #             # Segment the object
    #             sam2_predictor.set_image(camera_color_data)
    #             self.logger.info(f"bbox: {np.array([[x, y, x2, y2]])}")
    #             masks, scores, logits = sam2_predictor.predict(
    #                 point_coords=None,
    #                 point_labels=None,
    #                 box=np.array([[x, y, x2, y2]]),
    #                 multimask_output=False,
    #             )
    #             if masks.ndim == 4:
    #                 masks = masks.squeeze(1)

    #             mask = masks[0] > 0.5
    #             mask = masks[0].astype(bool)
    #             camera_color_data[mask] = [0, 255, 0]

    #              # Find the centroid of the mask
    #             M = cv2.moments(mask.astype(np.uint8))
    #             if M["m00"] > 0:
    #                 cx = int(M["m10"] / M["m00"])
    #                 cy = int(M["m01"] / M["m00"])
    #                 centroid = (cx, cy)
    #                 self.logger.info(f"Centroid: {centroid}")
    #                 depth_value = camera_depth_data[cy, cx] / 1000.0

    #                 # Publish the centroid
    #                 centroid_msg = VisualServoData(x=cx, y=cy, depth=depth_value)
    #                 #self.visual_servo_pub.publish(centroid_msg)

    #         cv2.imshow('vis', camera_color_data)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     cv2.destroyAllWindows()

    def track_item(self, bbox):
        # Initialize SORT tracker
        tracker = Sort()

        # Get initial camera frame
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()

        # Extract bounding box coordinates
        x, y, x2, y2 = map(int, bbox.xyxy[0])
        width, height = x2 - x, y2 - y
        self.logger.debug(f"Initial bbox: {(x, y, width, height)}")

        while True:
            # Get new camera frame
            camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()

            # Convert bbox to (x1, y1, x2, y2, confidence) format for SORT
            detections = np.array([[x, y, x2, y2, 1.0]])  # 1.0 is confidence score

            # Update tracker with detections
            tracked_objects = tracker.update(detections)

            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, obj)

                # Draw tracking box
                cv2.rectangle(camera_color_data, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(camera_color_data, f'ID: {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show frame with tracking
            cv2.imshow('vis', camera_color_data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


    def clear_plate(self, cup=False):
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        if camera_color_data is None:
            self.logger.warning("No camera data received.")
            return
        
        if not cup:
            self.logger.debug("Not a cup.")
            asyncio.run(self.robot_controller.reset())
            food_items = self.recognize_items(camera_color_data)
            #food_items = ["chicken tenders"]
            food_items = raf_utils.randomize_selection(food_items)
            
            category = raf_utils.get_category_from_label(food_items)
            det_prompt = self.food_selection(food_items)
            
        else:
            asyncio.run(self.robot_controller.move_to_cup_joint())
            det_prompt = 'cup .'

        # Prompt DINO-X with the detection prompt
        annotated_frame, detection, mask, label = self.detect_food(camera_color_data, det_prompt)
        #self.track_item(detection)
        if annotated_frame is None or detection is None or mask is None or label is None:
            self.logger.warning("food item not detected. Terminating Script")
            self.clear_plate()
            return
        if not cup:
            if not self.AUTONOMY:
                self.logger.info(f"labels: {label}")
                self.get_manual_action(annotated_frame, detection, mask, label, food_items, category)
            elif self.AUTONOMY:
                self.logger.info(f"labels: {label}")
                self.get_autonomy_action(annotated_frame, detection, mask, label, food_items, category)
        else:
            self.logger.info(f"labels: {label}")
            self.logger.debug("Cup detected.")
            self.get_cup_action(label)


    def get_manual_action(self, annotated_frame, detection, food_mask, label, food_items, category, insurance=0.985, close=1.05):
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        self.logger.info(f"detection: {detection}")
        self.logger.debug("category: %s", category)
        self.show_image(annotated_frame, "Are the detected items correct? ")
        # Check voice command stack
        if self.get_command() == 'stop':
            self.clear_stack()
            sys.exit(1)
            return
        elif self.get_command() == 'drink':
            self.clear_stack()
            self.clear_plate(cup=True)
            return
        
        # Item selected for grasping
        grasp_point, centroid, yaw_angle, wp1, wp2, p1, p2 = self.calculate_grasp_point_width(food_mask, category)
        vis2 = self.draw_points(camera_color_data, grasp_point, p1, p2, wp1, wp2)
        self.show_image(vis2, "Is this the correct grasp point? ")

        pose, width_point1, width_point2, center_point = self.get_object_position(camera_info_data, camera_depth_data, yaw_angle, grasp_point, wp1, wp2, category)
        print('THIS IS THE CENTER POINT', center_point)
        grip_val = self.skill_library.getGripperWidth(width_point1, width_point2, insurance, close)
        asyncio.run(self.robot_controller.setting_gripper_value(grip_val))

        close, food_height = raf_utils.find_gripper_values(food_items,self.PLATE_DEPTH,center_point)

 
        hover_offset = 0.01
        pose.position.z += hover_offset
        asyncio.run(self.robot_controller.move_to_pose(pose))
        if not raf_utils.validate_with_user("Is the robot in the correct position? "):
            sys.exit(1)
            
        #added to change how far down the gripper goes for bowl snacks
        pose.position.z -= food_height + hover_offset
        asyncio.run(self.robot_controller.move_to_pose(pose))
        grasp = grip_val*close
        if grasp>1:
            grasp = 1
        asyncio.run(self.robot_controller.setting_gripper_value(grasp))
        if not raf_utils.validate_with_user("Did the robot grasp the object? "):
            pose.position.z += 0.1
            asyncio.run(self.robot_controller.move_to_pose(pose))
            self.clear_plate()  

        pose.position.z += 0.15     
        asyncio.run(self.robot_controller.move_to_pose(pose))
        if not self.isObjectGrasped():
            asyncio.run(self.robot_controller.reset())
            self.isRetryAttempt = True
            self.clear_plate()
            return
        if category == 'single-bite' and self.bite_transfer_height == 'TALL':
            asyncio.run(self.robot_controller.move_to_feed_pose('TALL'))
        elif category == 'single-bite' and self.bite_transfer_height == 'SHORT':
            asyncio.run(self.robot_controller.move_to_feed_pose('SHORT'))
        elif category == 'multi-bite' and self.bite_transfer_height == 'TALL':
            asyncio.run(self.robot_controller.move_to_multi_bite_transfer('TALL'))
        elif category == 'multi-bite' and self.bite_transfer_height == 'SHORT':
            asyncio.run(self.robot_controller.move_to_multi_bite_transfer('SHORT'))
        # Check for multi-bite
        while self.checkObjectGrasped():
            pass
        if not raf_utils.validate_with_user("Continue feeding? (y/n): "):
            sys.exit(1)
        asyncio.run(self.robot_controller.reset())
        self.clear_plate()



    def get_autonomy_action(self, annotated_frame, detection, food_mask, label, food_items, category, insurance=0.985, close=1.05):
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        if self.get_command() == 'stop':
            self.clear_stack()
            sys.exit(1)
            return
        elif self.get_command() == 'drink':
            self.clear_stack()
            self.clear_plate(cup=True)
            return
    
         # Item selected for grasping
        grasp_point, centroid, yaw_angle, wp1, wp2, p1, p2 = self.calculate_grasp_point_width(food_mask, category)
        pose, width_point1, width_point2, center_point = self.get_object_position(camera_info_data, camera_depth_data, yaw_angle, grasp_point, wp1, wp2, category)
        grip_val = self.skill_library.getGripperWidth(width_point1, width_point2, insurance, close)
        asyncio.run(self.robot_controller.setting_gripper_value(grip_val))
        #rospy.sleep(2)
        self.logger.debug(f"Gripper Width: {grip_val}")
        close, food_height = raf_utils.find_gripper_values(food_items,self.PLATE_DEPTH,center_point)  
        hover_offset = 0.01
        # go right above the food in the correct orientation
        pose.position.z += hover_offset
        asyncio.run(self.robot_controller.move_to_pose(pose))
        # moves down proper amount to grasp food
        pose.position.z -= food_height + hover_offset
        asyncio.run(self.robot_controller.move_to_pose(pose))
        grasp = grip_val*close
        if grasp>1:
            grasp = 1
        asyncio.run(self.robot_controller.setting_gripper_value(grasp))
        # move up with grasped food
        pose.position.z += 0.15     
        asyncio.run(self.robot_controller.move_to_pose(pose))

        # we need to straighten the end effector to ensure that the camera cable doesnt get tangled
        #pose.orientation.something = something
        #self.robot_controller.move_to_pose(pose)
        
      
        if not self.isObjectGrasped():
            asyncio.run(self.robot_controller.reset())
            self.isRetryAttempt = True
            asyncio.run(self.robot_controller.setting_gripper_value(grasp - 0.2))
            self.clear_plate()
            return
        if category == 'single-bite' and self.bite_transfer_height == 'TALL':
            asyncio.run(self.robot_controller.move_to_feed_pose('TALL'))
        elif category == 'single-bite' and self.bite_transfer_height == 'SHORT':
           asyncio.run(self.robot_controller.move_to_feed_pose('SHORT'))
        elif category == 'multi-bite' and self.bite_transfer_height == 'TALL':
            asyncio.run(self.robot_controller.move_to_multi_bite_transfer('TALL'))
        elif category == 'multi-bite' and self.bite_transfer_height == 'SHORT':
            asyncio.run(self.robot_controller.move_to_multi_bite_transfer('SHORT'))
        while self.checkObjectGrasped():
            pass
        asyncio.run(self.robot_controller.reset())
        self.clear_plate()


    
    def get_cup_action(self, det_prompt, insurance=0.985, close=1.05):
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        annotated_frame, detection, mask, label = self.detect_food(camera_color_data, det_prompt)
        centroid = self.calculate_grasp_point_width(mask, 'drink')
        self.logger.info(f"Centroid: {centroid}")
        vis2 = self.draw_points(camera_color_data, centroid)
        if not self.AUTONOMY:
            self.show_image(vis2, "Is this the correct grasp point? ")
        pose = self.get_object_position(camera_info_data, camera_depth_data, None, centroid, None, None, 'drink')
        self.logger.debug(f"Pose: {pose}")
        asyncio.run(self.robot_controller.setting_gripper_value(0.0))
        cup_position = copy.deepcopy(pose)
        asyncio.run(self.robot_controller.move_to_pose(pose))
        if not self.AUTONOMY:
            if not raf_utils.validate_with_user("Is the robot in the correct position? "):
                sys.exit(1)
            asyncio.run(self.robot_controller.setting_gripper_value(0.65))
            if not raf_utils.validate_with_user("Did the robot grasp the object? "):
                pose.position.z += 0.1
                asyncio.run(self.robot_controller.move_to_pose(pose))
                self.clear_plate()
        else:
            asyncio.run(self.robot_controller.setting_gripper_value(0.65))
        pose.position.z += 0.15
        asyncio.run(self.robot_controller.move_to_pose(pose))
        asyncio.run(self.robot_controller.move_to_sip_pose(self.bite_transfer_height))
        time.sleep(5)
        asyncio.run(self.robot_controller.move_to_cup_joint())
        asyncio.run(self.robot_controller.move_to_pose(cup_position))
        asyncio.run(self.robot_controller.setting_gripper_value(0))
        cup_position.position.z += 0.25
        asyncio.run(self.robot_controller.move_to_pose(cup_position))
        asyncio.run(self.robot_controller.reset())
        self.clear_plate()
                
    
    
    def show_image(self, image, user_validation="Image"):
        cv2.imshow('image', image)
        cv2.waitKey(0)
        if not raf_utils.validate_with_user(user_validation):
            sys.exit(1)
        cv2.destroyAllWindows()







    def calculate_grasp_point_width(self, item_mask, category):
        if category == 'multi-bite':
            centroid = detect_centroid(item_mask)
            # get the coordinates of midpoints and points for width calculation
            p1,p2,width_p1,width_p2,box = raf_utils.get_box_points(item_mask)
            self.logger.debug(f"Centroid: {centroid}")

            # finds orientation of food
            yaw_angle = raf_utils.pretzel_angle_between_pixels(p1,p2)
            self.logger.debug(f"Yaw Angle: {yaw_angle}")

            # find bottom of food (on the segment)
            lower_center = detect_lower_center(item_mask)
            self.logger.debug(f"Lower Center: {lower_center}")
            
            # finds the midpoint between center and bottom of food
            grasp_point = get_grasp_points(centroid, lower_center)
            self.logger.debug(f"Grasp Point: {grasp_point}")

            # find the points on the box across from the grasp point
            gp1,gp2 = raf_utils.get_width_points(grasp_point, item_mask) # get the correct width for the gripper based on the mask
            self.logger.debug(f"Grasp Point 1: {gp1}")
            

            # find the nearest point on the mask to the points
            wp1,wp2 = mask_width_points(gp1,gp2,item_mask)  
            self.logger.debug(f"Width Point 1: {wp1}")    

            return grasp_point, centroid, yaw_angle, wp1, wp2, p1, p2
        
        elif category == 'single-bite':
            centroid = detect_centroid(item_mask)
            p1,p2,width_p1,width_p2,box = raf_utils.get_box_points(item_mask)
            yaw_angle = raf_utils.pretzel_angle_between_pixels(p1,p2)
            lower_center = detect_lower_center(item_mask)
            grasp_point = get_grasp_points(centroid, lower_center)
            gp1,gp2 = raf_utils.get_width_points(centroid, item_mask) # get the correct width for the gripper based on the mask
            wp1,wp2 = mask_width_points(gp1,gp2,item_mask)

            return centroid, grasp_point, yaw_angle, wp1, wp2, p1, p2
        
        elif category == 'drink':
            centroid = detect_centroid(item_mask)
            # p1,p2,width_p1,width_p2,box = raf_utils.get_cup_box_points(item_mask)
            # far_right_point, far_left_point, centroid1, mask_with_points = raf_utils.get_cup_box_points_v2(item_mask)
            # yaw_angle = raf_utils.pretzel_angle_between_pixels(p1,p2)
            # lower_center = detect_lower_center(item_mask)
            # grasp_point = get_grasp_points(centroid, lower_center) 

            return centroid
        

    def draw_points(self, image, center, lower=None, mid=None, wp1=None, wp2=None, box=None):
        if box is not None:
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        if wp1 is not None and wp2 is not None:
            cv2.line(image, wp1, wp2, (255,0,0), 2)

        cv2.circle(image, center, 5, (0,255,0), -1)

        if lower is not None and mid is not None:
            cv2.circle(image, lower, 5, (0,0,255), -1)
            cv2.circle(image, mid, 5, (255,0,0), -1)
        return image  
    

    def get_object_position(self, camera_info_data, camera_depth_data, yaw_angle, grasp_point, wp1, wp2, category):
        if category == 'drink':
            angle_of_rotation = 180
        else:
            angle_of_rotation = (180 - yaw_angle) - 30

        rot = self.get_rotation_matrix(radians(angle_of_rotation)) 
        validity, center_point = raf_utils.pixel2World(camera_info_data, grasp_point[0], grasp_point[1], camera_depth_data)
        print("Center Point: ", center_point)
         # for the width calculation
        if category != 'drink':
            validity, width_point1 = raf_utils.pixel2World(camera_info_data, wp1[0], wp1[1], camera_depth_data)
            validity, width_point2 = raf_utils.pixel2World(camera_info_data, wp2[0], wp2[1], camera_depth_data)
            
        if not validity:
            self.logger.error("Invalid world coordinates for centroid.")
            sys.exit(1)
        if center_point is None and category == 'drink':
            self.logger.error("No depth data! Retrying...")
            self.clear_plate(cup=True)
            return
        
        if category == 'drink':
            center_point[2] += 0.10


        food_transform = np.eye(4)
        food_transform[:3,3] = center_point.reshape(1,3)
        food_transform[:3,:3] = rot
        food_base = self.tf_utils.getTransformationFromTF("base_link", "camera_link") @ food_transform
        pose = self.tf_utils.get_pose_msg_from_transform(food_base)
        pose.position.y += self.CAMERA_OFFSET # Realsense camera offset
        pose.position.z -= self.Z_OFFSET # 0.01
        pose.position.x += 0.002 # Tilt of realsense to kinova moves desire pos down to 0.005
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

        if category != 'drink':
            return pose, width_point1, width_point2, center_point
        else:
            return pose



    def get_rotation_matrix(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]]) 




    def isObjectGrasped(self):
        camera_header, camera_color_data, camera_info_data, camera_depth_data = self.camera.get_camera_data()
        x = 805
        y = 633
        validity, points = raf_utils.pixel2World(camera_info_data, x, y, camera_depth_data)
        if points is not None:
            points = list(points)
            print("Object Depth", points[2])
            # if points[2] > 0.200 and points[2] < 0.300:
            if points[2] < 0.200:
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