#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 00:55:23 2024

@author: surajit
"""

import json
import os
import config

main_Dir = config.main_dir

json_file_path = f'{main_Dir}car_and_person_Detection/data/raw_data/annotations/bbox-annotations.json'  
images_folder = f'{main_Dir}car_and_person_Detection/data/raw_data/images/'  
labels_folder = f'{main_Dir}car_and_person_Detection/data/raw_data/labels/'  

with open(json_file_path) as f:
    data = json.load(f)

os.makedirs(labels_folder, exist_ok=True)

categories = {category['id']: category['name'] for category in data['categories']}
num_classes = len(categories)

def create_yolo_labels():
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']  
        category_id = annotation['category_id']

        image_info = next((img for img in data['images'] if img['id'] == image_id), None)
        if image_info:
            file_name = image_info['file_name']
            label_file_path = os.path.join(labels_folder, f"{file_name.split('.')[0]}.txt")

            img_width = image_info['width']
            img_height = image_info['height']
            
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height

            with open(label_file_path, 'a') as label_file:
                label_file.write(f"{category_id - 1} {x_center} {y_center} {width} {height}\n")  
                
create_yolo_labels()

/home/surajit/d/submission/car_and_person_Detection/data/raw_data/annotations/
/home/surajit/d/submission/car_and_person_Detection/data/raw_data/annotations/
/home/surajit/d/submission/data/raw_data/annotations/bbox-annotations.json
