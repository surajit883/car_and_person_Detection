#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:45:33 2024

@author: surajit
"""

from ultralytics import YOLO
import config

main_Dir = config.main_dir
model_path = f'{main_Dir}car_and_person_Detection/models/'
model = YOLO(model_path)  

model.export(format="onnx", opset=9, dynamic=True, simplify=True)


