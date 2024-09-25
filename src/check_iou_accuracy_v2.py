#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:19:26 2024

@author: surajit
"""

import os
from PIL import Image


def read_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size  


def read_ground_truth_label_file(file_path, image_width, image_height):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:  
                class_id = int(parts[0])
                x_center = float(parts[1]) * image_width
                y_center = float(parts[2]) * image_height
                width = float(parts[3]) * image_width
                height = float(parts[4]) * image_height

                
                x1 = x_center - (width / 2)
                y1 = y_center - (height / 2)
                x2 = x_center + (width / 2)
                y2 = y_center + (height / 2)

                boxes.append((class_id, [x1, y1, x2, y2]))
    return boxes


def read_predictions_label_file(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 1:
                image_id = parts[0].strip()  
                try:
                    class_id = int(float(parts[1].strip()))  
                except ValueError as e:
                    print(f"Error converting class ID: {parts[1]} - {e}")
                    continue  
                bbox = list(map(float, map(str.strip, parts[2:6]))) 
                boxes.append((image_id, class_id, bbox))
    return boxes


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_predictions(pred_path, ground_truth_folder, iou_threshold=0.5):
    class_correct = {}
    class_total = {}
    class_true_positive = {}
    class_false_positive = {}
    overall_correct = 0

    predictions_path = os.path.join(pred_path, 'predictions.txt')
    predicted_boxes = read_predictions_label_file(predictions_path)

    predictions_dict = {}
    for image_id, class_id, bbox in predicted_boxes:
        if image_id not in predictions_dict:
            predictions_dict[image_id] = []
        predictions_dict[image_id].append((class_id, bbox))

    for filename in os.listdir(ground_truth_folder):
        if filename.endswith('.txt'):
            image_folder = '/home/surajit/d/assigment_v2/data/test/images/'
            image_name = filename.replace('.txt', '.jpg')  
            image_path = os.path.join(image_folder, image_name)
            image_width, image_height = read_image_dimensions(image_path)
            gt_path = os.path.join(ground_truth_folder, filename)
            ground_truth_boxes = read_ground_truth_label_file(gt_path, image_width, image_height)

            gt_matched = set()  

            image_id = filename[:-4] + '.jpg'  
            if image_id in predictions_dict.keys():
                for pred_class, pred_bbox in predictions_dict[image_id]:
                    matched = False
                    for gt_index, (gt_class, gt_bbox) in enumerate(ground_truth_boxes):
                        if gt_index in gt_matched:
                            continue  
                        
                        if pred_class == gt_class:
                            iou = calculate_iou(pred_bbox, gt_bbox)
                            print(f"IoU: {iou} for Predicted Class: {pred_class} and Ground Truth Class: {gt_class}")
                            if iou >= iou_threshold:  
                                class_correct[pred_class] = class_correct.get(pred_class, 0) + 1
                                class_true_positive[pred_class] = class_true_positive.get(pred_class, 0) + 1
                                class_total[pred_class] = class_total.get(pred_class, 0) + 1
                                overall_correct += 1
                                gt_matched.add(gt_index)  
                                matched = True
                                break
                    if not matched:
                        class_false_positive[pred_class] = class_false_positive.get(pred_class, 0) + 1

            for gt_index, (gt_class, _) in enumerate(ground_truth_boxes):
                if gt_index not in gt_matched:
                    class_total[gt_class] = class_total.get(gt_class, 0) + 1

    class_accuracy = {cls: (class_correct.get(cls, 0) / class_total[cls] * 100) if cls in class_total else 0
                      for cls in class_total.keys()}
    
    class_precision = {cls: (class_true_positive.get(cls, 0) / (class_true_positive.get(cls, 0) + class_false_positive.get(cls, 0)) * 100) if (class_true_positive.get(cls, 0) + class_false_positive.get(cls, 0)) > 0 else 0
                       for cls in class_total.keys()}

    overall_accuracy = (overall_correct / sum(class_total.values())) * 100 if sum(class_total.values()) > 0 else 0

    return class_accuracy, class_precision, overall_accuracy

ground_truth_folder = '/home/surajit/d/assigment_v2/data/test/labels/'  
predictions_folder = '/home/surajit/d/assigment_v2/data/test/predicted/label/'  

class_accuracy, class_precision, overall_accuracy = evaluate_predictions(predictions_folder, ground_truth_folder)

print("Per-Class Accuracy:")
for cls, acc in class_accuracy.items():
    print(f"Class {cls}: {acc:.2f}%")

print("Per-Class Precision:")
for cls, prec in class_precision.items():
    print(f"Class {cls}: {prec:.2f}%")

print(f"Overall Accuracy: {overall_accuracy:.2f}%")
