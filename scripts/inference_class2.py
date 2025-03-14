from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt

import os
import cv2
import json
import torch
import tempfile
import sys
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

PATH_TO_GROUNDED_SAM2 = '/home/labuser/raf_v3_ws/Grounded-SAM-2'
API_TOKEN = "2b619f1c4b7434549812bae4690e52d8"
TEXT_PROMPT = "car . building ."
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

class inference_class2:
    
    def __init__(self):
        self.api_token = API_TOKEN
        self.text_prompt = TEXT_PROMPT
        self.img_path = IMG_PATH
        self.sam2_checkpoint = SAM2_CHECKPOINT
        self.sam2_model_config = SAM2_MODEL_CONFIG
        self.box_threshold = BOX_THRESHOLD
        self.with_slice_inference = WITH_SLICE_INFERENCE
        self.slice_wh = SLICE_WH
        self.overlap_ratio = OVERLAP_RATIO
        self.device = DEVICE
        self.output_dir = OUTPUT_DIR
        self.dump_json_results = DUMP_JSON_RESULTS
        
        
    def detection_values(self):
        config = Config(self.api_token)
        client = Client(config)
        classes = [x.strip().lower() for x in self.text_prompt.split('.') if x]
        class_name_to_id = {name: id for id, name in enumerate(classes)}
        class_id_to_name = {id: name for name, id in class_name_to_id.items()}

        image_url = client.upload_file(self.img_path)
        task = DinoxTask(
            image_url=image_url,
            prompts=[TextPrompt(text=self.text_prompt)],
            bbox_threshold=self.box_threshold,
            targets=[DetectionTarget.BBox],
        )
        client.run_task(task)
        result = task.result
        print(result)

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
        self.sam2_prediction(input_boxes, confidences, class_names, class_ids)
    
    def sam2_prediction(self, input_boxes, confidences, class_names, class_ids):
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)

        image = Image.open(self.img_path)
        sam2_predictor.set_image(np.array(image.convert("RGB")))
        
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        self.visualize_model(masks, scores, logits, confidences, class_names, class_ids, input_boxes)
    
    def visualize_model(self, masks, scores, logits, confidences, class_names, class_ids, input_boxes):

        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        labels = [
            f"{class_name}: {confidence:.2f}"
            for class_name, confidence in zip(class_names, confidences)
        ]

        img = cv2.imread(self.img_path)
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
        )

        print("Detections: ", detections)
        print("Labels: ", labels)
        print("Masks: ", masks[1])
        print("Scores: ", np.argmax(scores))
        

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "dinox_annotated_image.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "dinox_sam2_annotated_image_with_mask.jpg"), annotated_frame)

        print(f'Annotated image has already been saved as to "{OUTPUT_DIR}"')