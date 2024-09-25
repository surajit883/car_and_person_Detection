#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:32:46 2024

@author: surajit
"""

from ultralytics import YOLO

model_path = 'yolov8n.pt'  
data_yaml = '/data1/from7/satyukt/Projects/detectron2/trainval/data.yaml'  
epochs = 50  
img_size = 640  
batch_size = 8  
name = 'yolov8_experiment'  

model = YOLO(model_path)

model.train(data=data_yaml, epochs=epochs, imgsz=img_size, batch=batch_size, name=name)

