#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 01:10:09 2024

@author: surajit
"""

import os
import random
import shutil
import config


main_Dir = config.main_dir

base_dir = f'{main_Dir}car_and_person_Detection/data/'  
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(directory, 'images'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'labels'), exist_ok=True)


image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]


random.shuffle(image_files)


train_size = 0.8
val_size = 0.1
test_size = 0.1

num_images = len(image_files)
num_train = int(num_images * train_size)
num_val = int(num_images * val_size)


train_images = image_files[:num_train]
val_images = image_files[num_train:num_train + num_val]
test_images = image_files[num_train + num_val:]


def copy_files(image_list, source_images_dir, source_labels_dir, target_images_dir, target_labels_dir):
    for image in image_list:
        
        shutil.copy(os.path.join(source_images_dir, image), os.path.join(target_images_dir, image))
        
        label_file = image.replace('.jpg', '.txt')
        shutil.copy(os.path.join(source_labels_dir, label_file), os.path.join(target_labels_dir, label_file))


copy_files(train_images, images_dir, labels_dir, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'))
copy_files(val_images, images_dir, labels_dir, os.path.join(val_dir, 'images'), os.path.join(val_dir, 'labels'))
copy_files(test_images, images_dir, labels_dir, os.path.join(test_dir, 'images'), os.path.join(test_dir, 'labels'))

print(f"Dataset split into {len(train_images)} training, {len(val_images)} validation, and {len(test_images)} testing images.")
