import os
import sys
import pickle
import random
import timeit
from tqdm import tqdm
import cv2

import numpy as np
import tensorflow as tf
from preprocess_data import preprocess_data, preprocess_data_validation, preprocess_data_test

from config import num_classes, network_size, anchor_boxes
from config import train_dict_address, validation_dict_address


test_data_generator_folder = "test_data_generator/"

if not os.path.exists(test_data_generator_folder):
    os.mkdir(test_data_generator_folder)
    
if len(os.listdir(test_data_generator_folder)) > 0:
    os.system('rm -r ' + test_data_generator_folder)
    os.mkdir(test_data_generator_folder)
    
# CREATE TF DATA DATASET WITH MAP AND MULTI-THREAD READING
train_data_information = pickle.load(open(train_dict_address, "rb"))
random.shuffle(train_data_information)
train_data = []
train_labels = []

for data, label, augment_type in train_data_information:
    train_data.append(os.path.join(data))
    train_labels.append(np.array(label).astype('float32'))

train_labels = tf.ragged.constant(train_labels)

print("Total Number of Final Train Images = " + str(len(train_data)))

"""
autotune = tf.data.experimental.AUTOTUNE
batch_size = 8
# CREATE TRAIN DATASET
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_dataset = train_dataset.cache().shuffle(len(train_data))
train_dataset = train_dataset.map(preprocess_data_test, num_parallel_calls=autotune)
train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
train_dataset = train_dataset.prefetch(autotune)
"""

# NOW WE CAN TEST OUR DATASETS
image_index = 1
for data, label, augment_type in train_data_information:
    label = np.array(label).astype('float32')
    image, label, decoded_labels = preprocess_data_test(data, label)
    print(image.shape)
    print(label.shape)
    print(decoded_labels.shape)
    gt_color = (0, 0, 255)
    color = (255, 0, 0)
    image = image.numpy() * 255.
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    indices = decoded_labels[:, -1] >= 0
    
    for i in indices:
        if bbox[-1] >= 0:
            x_min = (anchor_boxes[i][0] - anchor_boxes[i][2] / 2)
            y_min = (anchor_boxes[i][1] - anchor_boxes[i][3] / 2)
            x_max = (anchor_boxes[i][0] + anchor_boxes[i][2] / 2)
            y_max = (anchor_boxes[i][1] + anchor_boxes[i][3] / 2)
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness=1)
            
    for i, bbox in enumerate(label):
        x_min = (bbox[0] - bbox[2] / 2)
        y_min = (bbox[1] - bbox[3] / 2)
        x_max = (bbox[0] + bbox[2] / 2)
        y_max = (bbox[1] + bbox[3] / 2)
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), gt_color, thickness=2)
                
    cv2.imwrite(test_data_generator_folder + data.split("/")[-2] + "_" + data.split("/")[-1], image)
    image_index += 1        
        
    print("---------------------------------------")










