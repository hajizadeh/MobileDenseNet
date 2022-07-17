import numpy as np
import os
import random
from xml.etree import ElementTree
import pickle
import json
import itertools


class XMLTimeDataLoader:

    def __init__(self, json_file_address, images_prefix_address, shuffle=True, read_objects_from_file=False, only_wanted_objects=False):

        self.json_file_address = json_file_address
        self.images_prefix_address = images_prefix_address
        
        self.json_file = json.load(open(self.json_file_address))
        print("Number of Files = " + str(len(self.json_file['images'])))
        print("Number of Categories = " + str(len(self.json_file['categories'])))
        print("Number of Annotations = " + str(len(self.json_file['annotations'])))
        
        if only_wanted_objects:
            #self.only_wanted_objects_list = ['person', 'kitchen', 'food', 'furniture', 'tv', 'laptop', 'mouse',
            #                                'keyboard', 'cell phone', 'sink', 'book', 'clock', 'vase', 'teddy bear']
            self.only_wanted_objects_list = ['person']
            
        self.only_wanted_objects = only_wanted_objects            

        if not read_objects_from_file:
            self.coco_object_names = {}
            self.object_names = {}
            self.find_all_object_names('coco_real_object_names.pkl', 'object_names_coco.pkl')     
            
        else:
            self.coco_object_names = pickle.load(open('coco_real_object_names.pkl', "rb"))
            self.object_names = pickle.load(open('object_names_coco.pkl', "rb"))  
        
        self.image_name_width_height = {}        
        self.find_image_name_width_height()

        self.shuffle = shuffle
        self.num_classes = len(self.object_names)
        self.num_images = len(self.image_name_width_height)

    def find_all_object_names(self, object_real_name_address, object_dict_address):
    
        value = 0
    
        for category in self.json_file['categories']:        
            self.coco_object_names[category['id']] = [category['name'], category['supercategory']]
            
            if not category['name'] in self.object_names and not category['supercategory'] in self.object_names:
                if self.only_wanted_objects:
                    if category['name'] in self.only_wanted_objects_list:
                        self.object_names[category['name']] = value
                        value += 1
                    elif category['supercategory'] in self.only_wanted_objects_list:
                        self.object_names[category['supercategory']] = value
                        value += 1
                else:
                    self.object_names[category['name']] = value
                    # self.object_names[category['supercategory']] = value
                    value += 1
                    

        print("All the objects found in folder is = " + str(len(self.coco_object_names)))
        print(self.coco_object_names)
        pickle.dump(self.coco_object_names, open(object_real_name_address, 'wb'))

        print("All the objects found in folder is = " + str(len(self.object_names)))
        print(self.object_names)
        pickle.dump(self.object_names, open(object_dict_address, 'wb'))        
        
        return self.object_names
        
    def find_image_name_width_height(self):    
        for image in self.json_file['images']:        
            self.image_name_width_height[image['id']] = {'file_name': image['file_name'], 'width': image['width'], 'height': image['height']}            
        

    def create_data_list(self, output_filename):
    
        # The COCO bounding box format is [top left x position, top left y position, width, height].
    
        data_dict = {}
        khar_boxes = 0.0
        
        for annotation in self.json_file['annotations']:
        
            name_of_file = self.image_name_width_height[annotation['image_id']]['file_name']
            width = self.image_name_width_height[annotation['image_id']]['width']
            height = self.image_name_width_height[annotation['image_id']]['height']
            
            if self.coco_object_names[annotation['category_id']][0] in self.object_names:
                class_label = self.object_names[self.coco_object_names[annotation['category_id']][0]]                
                
            elif self.coco_object_names[annotation['category_id']][1] in self.object_names:
                class_label = self.object_names[self.coco_object_names[annotation['category_id']][1]]
                
            else:
                continue
            
            raw_bbox = annotation['bbox']
            if raw_bbox[2] < 5. or raw_bbox[3] < 5.:
                khar_boxes += 1.
                continue
                # print(name_of_file)
                # print(raw_bbox)
            bbox = [(raw_bbox[0] + raw_bbox[2]/2) / width, (raw_bbox[1] + raw_bbox[3]/2) / height, raw_bbox[2] / width, raw_bbox[3]/ height, class_label]
            
            if bbox[2] == 0.0 or bbox[3] == 0.0:
                khar_boxes += 1.
                continue
            
            # print(name_of_file)
            
            ground_truth_data = bbox
            # print(ground_truth_data)
            
            # print('-----------------')
            
            if name_of_file in data_dict:
                data_dict[name_of_file].append(ground_truth_data)
                # print(np.shape(data_dict[name_of_file]))
            else:
                data_dict[name_of_file] = [ground_truth_data]
        
        print("Length of data is = " + str(len(data_dict)))
        
        num_annotations_collected = 0
        for data, gt in data_dict.items():
            num_annotations_collected += len(gt)
            
        print("Length of annotations collected = " + str(num_annotations_collected))
        print("Length of khar annotations collected = " + str(khar_boxes))
        
        # Convert data_dict to data_list
        data_list = []
        for address, gt in data_dict.items():
            data_list.append([os.path.join(self.images_prefix_address, address), np.array(gt), 'none'])
        
        if self.shuffle:
            random.shuffle(data_list)

        # Data is a shuffled list of dictionaris
        pickle.dump(data_list, open(output_filename + '.pkl', 'wb'))
        
        for data in data_list[:10]:
            print(data[0])
            print(np.shape(data[1]))

        return data_list


if __name__ == '__main__':
    print("Data Reader is Main")
    
    # instances_valminusminival2014.json
    # instances_train2014.json
    
    dr = XMLTimeDataLoader('/home/mohammad.hajizadeh/object_detection_datasets/CoCo2017/annotations/instances_val2017.json',
                          '/home/mohammad.hajizadeh/object_detection_datasets/CoCo2017/val2017/',
                          read_objects_from_file=False)
    print("Object Names Collected")    
    data = dr.create_data_list('coco_val')
    
    dr = XMLTimeDataLoader('/home/mohammad.hajizadeh/object_detection_datasets/CoCo2017/annotations/instances_train2017.json',
                          '/home/mohammad.hajizadeh/object_detection_datasets/CoCo2017/train2017/',
                          read_objects_from_file=True)    
    data = dr.create_data_list('coco_train')


