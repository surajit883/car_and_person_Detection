#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:32:46 2024

@author: surajit
"""

from ultralytics import YOLO
import cv2
import os
import config


main_Dir = config.main_dir
model_path = f'{main_Dir}car_and_person_Detection/models/best.pt'
model = YOLO(model_path)

def predict_and_save_image(image_path, save_dir, name, output_label_file):
    predictions = []

    img = cv2.imread(image_path)
    results = model(img)

    if results:
        annotated_img = results[0].plot()  
        image_name = os.path.basename(image_path)  
        save_path = os.path.join(save_dir, f"predicted_{image_name}")  
        cv2.imwrite(save_path, annotated_img)  
        print(f"Saved predicted image at: {save_path}")
        
        boxes = results[0].boxes.xyxy  
        class_ids = results[0].boxes.cls  
        confidences = results[0].boxes.conf  

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = box.int() 

            predictions.append({
                'image': name,
                'class_id': class_id.item(),  
                'confidence': confidence.item(),  
                'bbox': [x1.item(), y1.item(), x2.item(), y2.item()]  
            })

        with open(output_label_file, 'a') as f:  
            for pred in predictions:
                bbox = pred['bbox']
                f.write(f"{pred['image']}, {pred['class_id']}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, {pred['confidence']}\n")
    
    else:
        print(f"No predictions for image: {name}")

image_dir = '/home/surajit/d/assigment_v2/test/images/'  
save_dir = '/home/surajit/d/assigment_v2/test/predicted/images/'  
save_label_dir = '/home/surajit/d/assigment_v2/test/predicted/label/'

os.makedirs(save_dir, exist_ok=True)
output_label_file = os.path.join(save_label_dir, 'predictions.txt')
os.makedirs(save_label_dir, exist_ok=True)  

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg')):  
        image_path = os.path.join(image_dir, filename)
        
        predict_and_save_image(image_path, save_dir, filename, output_label_file)
