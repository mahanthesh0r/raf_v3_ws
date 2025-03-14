#!/usr/bin/env python3
import cv2
import numpy as np
import math
import cmath

from scipy.spatial import Delaunay

import torch
from torchvision.transforms import ToTensor

from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy


def efficient_sam_box_prompt_segment(image, pts_sampled, model):
    bbox = torch.reshape(torch.tensor(pts_sampled), [1, 1, 2, 2])
    bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(image)

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].cuda(),
        bbox.cuda(),
        bbox_labels.cuda(),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
            curr_predicted_iou > max_predicted_iou
            or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou


def detect_plate(img, multiplier = 1.7):
    H,W,C = img.shape
    print("Detected plate H,W,C", H,W,C)
    img_orig = img.copy()
    img = cv2.resize(img, (W//2, H//2))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred,
                       cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                   param2 = 30, minRadius = 50, maxRadius = 200)
    plate_mask = np.zeros((H,W)).astype(np.uint8)
    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            plate_small_mask = plate_mask.copy()
            cv2.circle(plate_small_mask, (a*2, b*2), int(r*multiplier), (255,255,255), -1)
            plate_mask_vis = np.repeat(plate_mask[:,:,np.newaxis], 3, axis=2)
            break
        return plate_small_mask
    

def cleanup_mask(mask, blur_kernel_size=(5, 5), threshold=127, erosion_size=3):
    """
    Applies low-pass filter, thresholds, and erodes an image mask.

    :param image: Input image mask in grayscale.
    :param blur_kernel_size: Size of the Gaussian blur kernel.
    :param threshold: Threshold value for binary thresholding.
    :param erosion_size: Size of the kernel for erosion.
    :return: Processed image.
    """
    # Apply Gaussian Blur for low-pass filtering
    blurred = cv2.GaussianBlur(mask, blur_kernel_size, 0)
    # Apply thresholding
    _, thresholded = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    # Create erosion kernel
    erosion_kernel = np.ones((erosion_size, erosion_size), np.uint8)
    # Apply erosion
    eroded = cv2.erode(thresholded, erosion_kernel, iterations=1)
    return eroded

def mask_weight(mask):
    H,W = mask.shape
    return np.count_nonzero(mask)/(W*H)


def detect_centroid(mask):
    cX, cY = 0, 0
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
    centroid = proj_pix2mask(np.array([cX, cY]), mask)
    centroid = (int(centroid[0]), int(centroid[1]))
    return centroid


def detect_lower_center(mask):
    centroid = detect_centroid(mask)
    cX, cY = 0, 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Filter points that are below the centroid
        points_below_centroid = [point for point in largest_contour if point[0][1] > centroid[1]]
        # Find the furthest point in the largest contour from the reference point
        if points_below_centroid:
            furthest_point = max(points_below_centroid, key=lambda point: np.linalg.norm(np.array(point[0]) - np.array(centroid)))
            cX = furthest_point[0][0]
            cY = furthest_point[0][1]
    furthest_point = proj_pix2mask(np.array([cX, cY]), mask)
    furthest_point = (int(furthest_point[0]), int(furthest_point[1]))
    return furthest_point

# returns the points on the bounding box which are 1/3 the distance from the bottom of the food item
def get_grasp_points(center, lower_center):
    y_dist = abs(center[1] - lower_center[1])
    x_dist = abs(center[0] - lower_center[0])
    theta = math.atan2(y_dist, x_dist)
    # how much to move down the food item
    percent = 0.25
    dist = math.sqrt(x_dist**2 + y_dist**2)*percent
    x = dist*math.cos(theta)
    y = dist*math.sin(theta)
    final_y = int(center[1] + y)
    # x depends on angle of the food item
    if center[0] > lower_center[0]:
        final_x = int(center[0] - x)
    else:
        final_x = int(center[0] + x)
    
    #return (int((center[0] + lower_center[0]) / 2) , int((center[1] + lower_center[1]) / 2))

    return (final_x, final_y)
        



def mask_width_points(p1,p2,mask):
    p1 = np.array(p1)
    p2 = np.array(p2)
    wp1 = proj_pix2mask(p1, mask)
    wp2 = proj_pix2mask(p2, mask)

    return wp1, wp2



def proj_pix2mask(px, mask):
    ys, xs = np.where(mask > 0)
    if not len(ys):
        return px
    mask_pixels = np.vstack((xs,ys)).T
    neigh = NearestNeighbors()
    neigh.fit(mask_pixels)
    dists, idxs = neigh.kneighbors(np.array(px).reshape(1,-1), 1, return_distance=True)
    projected_px = mask_pixels[idxs.squeeze()]
    return projected_px