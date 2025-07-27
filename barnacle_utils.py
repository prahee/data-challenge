import streamlit as st 
import cv2
import numpy as np
import pandas as pd
import os
import random 

def load_image(image_path, grayscale=False):
    if not os.path.exists(image_path):
        st.error(f"Error: Image not found at {image_path}. Please check your file paths.")
        return None
    
    if grayscale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def extract_barnacle_features(mask_image, image_name=""):

    print(f"\n-> Starting to extract features from mask for {image_name}...")

    _, binary_outlines = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(binary_outlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"   Found {len(contours)} raw contours from outlines.")

    filled_mask = np.zeros_like(mask_image)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_mask, 8, cv2.CV_32S)
    
    print(f"   Found {num_labels - 1} connected components (excluding background) in the filled mask.")

    barnacle_data = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        centroid_x, centroid_y = centroids[i]
        bbox_x = stats[i, cv2.CC_STAT_LEFT]
        bbox_y = stats[i, cv2.CC_STAT_TOP]
        bbox_w = stats[i, cv2.CC_STAT_WIDTH]
        bbox_h = stats[i, cv2.CC_STAT_HEIGHT]

        component_only_mask = ((labels == i) * 255).astype(np.uint8)
        component_contours, _ = cv2.findContours(component_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        perimeter = 0
        circularity = 0
        aspect_ratio = 0
        solidity = 0
        
        if component_contours:
            main_contour = max(component_contours, key=cv2.contourArea) 
            
            perimeter = cv2.arcLength(main_contour, True)
            
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
            
            if bbox_h > 0:
                aspect_ratio = bbox_w / bbox_h
                
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
            barnacle_data.append({
                "image_name": image_name,
                "id_in_image": i,
                "area": float(area),
                "perimeter": float(perimeter),
                "circularity": float(circularity),
                "aspect_ratio": float(aspect_ratio),
                "solidity": float(solidity),
                "centroid_x": float(centroid_x),
                "centroid_y": float(centroid_y),
                "bbox_x": bbox_x, "bbox_y": bbox_y, 
                "bbox_w": bbox_w, "bbox_h": bbox_h,
                "contour": main_contour
            })
    print(f"   Extracted features for {len(barnacle_data)} barnacles (after filtering).")
    print(f"-> Feature extraction complete for {image_name}.")
    return barnacle_data