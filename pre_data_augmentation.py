import cv2
import os
import numpy as np
import pickle
import random
from shutil import copyfile
from tqdm import tqdm


def resize_image(image, value):
    resize_rand = (1 + np.random.rand()) / 2
    if np.random.rand() > 0.5:
        resized_image = cv2.resize(image, (int(image.shape[1] * resize_rand), image.shape[0]))
    else:
        resized_image = cv2.resize(image, (image.shape[1], int(image.shape[0] * resize_rand)))  
    
    return resized_image, value
    
    
def crop_image(image, value):
    crop_x_rand = np.random.rand() / 4.0
    crop_y_rand = np.random.rand() / 4.0
    if np.random.rand() > 0.5:
        croped_image = image[:, int(image.shape[1] * crop_x_rand / 2): int(image.shape[1] * (1 - crop_x_rand / 2)), :]
        
        value_copy = value.copy()
        new_value = value.copy()
        value_copy[:, 0] = value[:, 0] - value[:, 2] / 2
        value_copy[:, 1] = value[:, 1] - value[:, 3] / 2
        value_copy[:, 2] = value[:, 0] + value[:, 2] / 2
        value_copy[:, 3] = value[:, 1] + value[:, 3] / 2

        value_copy[:, 0] = np.clip((value_copy[:, 0] - crop_x_rand / 2) * (1 / (1 - crop_x_rand)), 0.0, 1.0)
        value_copy[:, 2] = np.clip((value_copy[:, 2] - crop_x_rand / 2) * (1 / (1 - crop_x_rand)), 0.0, 1.0)

        new_value[:, 0] = value_copy[:, 0] + (value_copy[:, 2] - value_copy[:, 0]) / 2
        new_value[:, 1] = value_copy[:, 1] + (value_copy[:, 3] - value_copy[:, 1]) / 2

        new_value[:, 2] = value_copy[:, 2] - value_copy[:, 0]
        new_value[:, 3] = value_copy[:, 3] - value_copy[:, 1]
    else:
        croped_image = image[int(image.shape[0] * crop_y_rand / 2): int(image.shape[0] * (1 - crop_y_rand / 2)), :, :]

        value_copy = value.copy()
        new_value = value.copy()
        value_copy[:, 0] = value[:, 0] - value[:, 2] / 2
        value_copy[:, 1] = value[:, 1] - value[:, 3] / 2
        value_copy[:, 2] = value[:, 0] + value[:, 2] / 2
        value_copy[:, 3] = value[:, 1] + value[:, 3] / 2

        value_copy[:, 1] = np.clip((value_copy[:, 1] - crop_y_rand / 2) * (1 / (1 - crop_y_rand)), 0.0, 1.0)
        value_copy[:, 3] = np.clip((value_copy[:, 3] - crop_y_rand / 2) * (1 / (1 - crop_y_rand)), 0.0, 1.0)

        new_value[:, 0] = value_copy[:, 0] + (value_copy[:, 2] - value_copy[:, 0]) / 2
        new_value[:, 1] = value_copy[:, 1] + (value_copy[:, 3] - value_copy[:, 1]) / 2

        new_value[:, 2] = value_copy[:, 2] - value_copy[:, 0]
        new_value[:, 3] = value_copy[:, 3] - value_copy[:, 1]
        
    final_value = []
    for box in new_value:
        if box[2] > 0.01 and box[3] > 0.01:
            final_value.append(box)
    
    return croped_image, np.array(final_value)
    

def random_zoom_in(image, value):
    crop_top_left_rand = np.random.rand() / 6.0
    crop_bot_right_rand = np.random.rand() / 6.0
    image_shape = np.shape(image)[:2]
    image = image[int(image_shape[0] * crop_top_left_rand):, int(image_shape[1] * crop_top_left_rand):, :]
    
    value_copy = value.copy()
    new_value = value.copy()
    value_copy[:, 0] = value[:, 0] - value[:, 2] / 2
    value_copy[:, 1] = value[:, 1] - value[:, 3] / 2
    value_copy[:, 2] = value[:, 0] + value[:, 2] / 2
    value_copy[:, 3] = value[:, 1] + value[:, 3] / 2
    
    value_copy[:, 0] = np.clip((value_copy[:, 0] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0)
    value_copy[:, 1] = np.clip((value_copy[:, 1] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0)
    value_copy[:, 2] = np.clip((value_copy[:, 2] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0)
    value_copy[:, 3] = np.clip((value_copy[:, 3] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0)
    
    image_shape = np.shape(image)[:2]
    image = image[:int(image_shape[0] * (1. - crop_bot_right_rand)), :int(image_shape[1] * (1. - crop_bot_right_rand)), :]
    
    value_copy[:, 0] = np.clip((value_copy[:, 0]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0)
    value_copy[:, 1] = np.clip((value_copy[:, 1]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0)
    value_copy[:, 2] = np.clip((value_copy[:, 2]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0)
    value_copy[:, 3] = np.clip((value_copy[:, 3]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0)
    
    new_value[:, 0] = value_copy[:, 0] + (value_copy[:, 2] - value_copy[:, 0]) / 2
    new_value[:, 1] = value_copy[:, 1] + (value_copy[:, 3] - value_copy[:, 1]) / 2

    new_value[:, 2] = value_copy[:, 2] - value_copy[:, 0]
    new_value[:, 3] = value_copy[:, 3] - value_copy[:, 1]
    
    final_value = []
    for box in new_value:
        if box[2] > 0.01 and box[3] > 0.01:
            final_value.append(box)
            
    return image, np.array(final_value)
    
    
def flip_horiz_image(image, value):
    flip_image = cv2.flip(image, 1)
    new_value = value.copy()
    new_value[:, 0] = 1. - value[:, 0]   
    
    return flip_image, new_value
    
    
def flip_vert_image(image, value):
    flip_image = cv2.flip(image, 0)
    new_value = value.copy()
    new_value[:, 1] = 1. - value[:, 1] 
    
    return flip_image, new_value


def change_hsv_setting_image(image, value):

    # SOME COLOR CHANGING IN HSV COLOR SPACE
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    h_rand = np.random.rand()
    s_rand = np.random.rand()
    v_rand = np.random.rand()

    if h_rand < 0.5:
        hsv_image[:, :, 0] = np.clip((hsv_image[:, :, 0] + (-0.25 + h_rand) * 60) % 180, 0, 179).astype(np.uint8)

    if s_rand < 0.6:
        hsv_image[:, :, 1] = np.clip((0.7 + s_rand) * hsv_image[:, :, 1], 0, 255).astype(np.uint8)

    if v_rand < 0.5:
        hsv_image[:, :, 2] = np.clip((hsv_image[:, :, 2] + (-0.25 + v_rand) * 128), 0, 255).astype(np.uint8)
        
    final_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    return final_image, value
    
    
def mix4_image(image, value, other_images):
    new_image = np.zeros((640, 640, 3), dtype=np.uint8)
    final_value = []
    new_value = value.copy()
    image = cv2.resize(image, (320, 320))
    new_image[:320, :320, :] = image
    
    new_value[:, 0] = value[:, 0] / 2
    new_value[:, 1] = value[:, 1] / 2
    new_value[:, 2] = value[:, 2] / 2
    new_value[:, 3] = value[:, 3] / 2
    
    for box in new_value:
        if box[2] > 0.01 and box[3] > 0.01:
            final_value.append(box)

    index = 0
    for key, value, augment_type in other_images:
        image = cv2.imread(key)
        image = cv2.resize(image, (320, 320))
        if index == 0:
            new_image[:320, 320:, :] = image
        elif index == 1:
            new_image[320:, :320, :] = image
        elif index == 2:
            new_image[320:, 320:, :] = image
        
        new_value = value.copy()
        new_value[:, 0] = value[:, 0] / 2
        new_value[:, 1] = value[:, 1] / 2
        new_value[:, 2] = value[:, 2] / 2
        new_value[:, 3] = value[:, 3] / 2
        if index == 0:
            new_value[:, 0] = new_value[:, 0] + 0.5
        elif index == 1:
            new_value[:, 1] = new_value[:, 1] + 0.5
        elif index == 2:
            new_value[:, 0] = new_value[:, 0] + 0.5
            new_value[:, 1] = new_value[:, 1] + 0.5
            
        for box in new_value:
            if box[2] > 0.01 and box[3] > 0.01:
                final_value.append(box)
                
        index+= 1
        
    return new_image, np.array(final_value)
    
    
def crop_mix4_image(image, value, other_images):
    new_image = np.zeros((640, 640, 3), dtype=np.uint8)
    final_value = []
    new_value = value.copy()
    image, new_value = random_zoom_in(image, new_value)
    image = cv2.resize(image, (320, 320))
    new_image[:320, :320, :] = image
    
    if len(new_value) > 0:
        new_value[:, 0] = new_value[:, 0] / 2
        new_value[:, 1] = new_value[:, 1] / 2
        new_value[:, 2] = new_value[:, 2] / 2
        new_value[:, 3] = new_value[:, 3] / 2
    
    for box in new_value:
        if box[2] > 0.01 and box[3] > 0.01:
            final_value.append(box)

    index = 0
    for key, value, augment_type in other_images:
        image = cv2.imread(key)
        image, value = random_zoom_in(image, value)
        image = cv2.resize(image, (320, 320))
        if index == 0:
            new_image[:320, 320:, :] = image
        elif index == 1:
            new_image[320:, :320, :] = image
        elif index == 2:
            new_image[320:, 320:, :] = image
        
        if len(value) > 0:
            new_value = value.copy()
            new_value[:, 0] = value[:, 0] / 2
            new_value[:, 1] = value[:, 1] / 2
            new_value[:, 2] = value[:, 2] / 2
            new_value[:, 3] = value[:, 3] / 2
            if index == 0:
                new_value[:, 0] = new_value[:, 0] + 0.5
            elif index == 1:
                new_value[:, 1] = new_value[:, 1] + 0.5
            elif index == 2:
                new_value[:, 0] = new_value[:, 0] + 0.5
                new_value[:, 1] = new_value[:, 1] + 0.5
            
        for box in new_value:
            if box[2] > 0.01 and box[3] > 0.01:
                final_value.append(box)
                
        index+= 1
        
    return new_image, np.array(final_value)
   

def create_more_data_using_augmentations(data_dict_address, destination_address, augmentated_data_dict_address):

    augmentations={'crop': False, 'resize': True, 'flip_horiz': True, 'flip_vert': False,
                  'hsv': True, 'hsv_flip_horiz': True, 'crop_flip_horiz': False, 'hsv_crop': False,
                  'hsv_crop_flip_horiz': False, 'mix4': True, 'crop_mix4': True}
                                            
    augmentated_pickle = []
    
    if not os.path.exists(destination_address):
        os.mkdir(destination_address)
    else:
        os.system('rm -r ' + destination_address)
        os.mkdir(destination_address)
    
    data_pickle = pickle.load(open(data_dict_address, "rb"))
    data_index = 0
    print("Length of data before augmentation is = " + str(len(data_pickle)))
    
    for key, value, augment_type in tqdm(data_pickle, desc="Augmenting Images:", bar_format='{desc} Step:{n_fmt}/{total_fmt} |{bar:30}|'):
        full_image_address = key      
        key = key[-35:].replace("/", "_")
        augmentated_pickle.append([os.path.join(destination_address, key), value, 'none'])
        # print(str(data_index) + ": " + full_image_address)
        data_index += 1
        copyfile(full_image_address, os.path.join(destination_address, key))
        image = cv2.imread(full_image_address)
        
        if augmentations['resize']:
            new_image, new_value = resize_image(image, value)
            new_key = key[:-5] + '_resize' + key[-5:]
            dest_image_address = os.path.join(destination_address, new_key)
            cv2.imwrite(dest_image_address, new_image)  
            augmentated_pickle.append([dest_image_address, new_value, 'resize'])
            
        if augmentations['crop']:
            new_image, new_value = crop_image(image, value)
            
            if len(new_value) > 0:
                new_key = key[:-5] + '_crop' + key[-5:]
                dest_image_address = os.path.join(destination_address, new_key)
                cv2.imwrite(dest_image_address, new_image)   
                augmentated_pickle.append([dest_image_address, new_value, 'crop'])
            else:
                print("ignored")
            
        if augmentations['flip_horiz']:
            new_image, new_value = flip_horiz_image(image, value)                    
            new_key = key[:-5] + '_flip_horiz' + key[-5:]
            dest_image_address = os.path.join(destination_address, new_key)
            cv2.imwrite(dest_image_address, new_image) 
            augmentated_pickle.append([dest_image_address, new_value, 'flip_horiz'])
            
        if augmentations['flip_vert']:
            new_image, new_value = flip_vert_image(image, value)
            
            new_key = key[:-5] + '_flip_vert' + key[-5:]
            dest_image_address = os.path.join(destination_address, new_key)
            cv2.imwrite(dest_image_address, new_image)   
            augmentated_pickle.append([dest_image_address, new_value, 'flip_vert'])
            
        if augmentations['hsv']:
            new_image, new_value = change_hsv_setting_image(image, value)            
                    
            new_key = key[:-5] + '_hsv' + key[-5:]
            dest_image_address = os.path.join(destination_address, new_key)
            cv2.imwrite(dest_image_address, new_image)
            augmentated_pickle.append([dest_image_address, new_value, 'hsv'])
            
        if augmentations['hsv_flip_horiz']:
            new_image, new_value = flip_horiz_image(image, value)
            new_image, new_value = change_hsv_setting_image(new_image, new_value)  
            
            new_key = key[:-5] + '_hsv_flip_horiz' + key[-5:]
            dest_image_address = os.path.join(destination_address, new_key)
            cv2.imwrite(dest_image_address, new_image)            
            augmentated_pickle.append([dest_image_address, new_value, 'hsv_flip_horiz'])
            
        if augmentations['crop_flip_horiz']:
            new_image, new_value = flip_horiz_image(image, value)
            new_image, new_value = crop_image(new_image, new_value)
            
            if len(new_value) > 0:
                new_key = key[:-5] + '_crop_flip_horiz' + key[-5:]
                dest_image_address = os.path.join(destination_address, new_key)
                cv2.imwrite(dest_image_address, new_image)   
                augmentated_pickle.append([dest_image_address, new_value, 'crop_flip_horiz'])
            else:
                print("ignored")
                
        if augmentations['hsv_crop']:
            new_image, new_value = change_hsv_setting_image(image, value) 
            new_image, new_value = crop_image(new_image, new_value)
            
            if len(new_value) > 0:
                new_key = key[:-5] + '_hsv_crop' + key[-5:]
                dest_image_address = os.path.join(destination_address, new_key)
                cv2.imwrite(dest_image_address, new_image)   
                augmentated_pickle.append([dest_image_address, new_value, 'hsv_crop'])
            else:
                print("ignored")
                
        if augmentations['hsv_crop_flip_horiz']:
            new_image, new_value = change_hsv_setting_image(image, value)
            new_image, new_value = flip_horiz_image(new_image, new_value)
            new_image, new_value = crop_image(new_image, new_value)
            
            if len(new_value) > 0:
                new_key = key[:-5] + '_hsv_crop_flip_horiz' + key[-5:]
                dest_image_address = os.path.join(destination_address, new_key)
                cv2.imwrite(dest_image_address, new_image)   
                augmentated_pickle.append([dest_image_address, new_value, 'hsv_crop_flip_horiz'])
            else:
                print("ignored")
        
        if augmentations['mix4']:
            three_other_images = random.sample(data_pickle, 3)
            new_image, new_value = mix4_image(image, value, three_other_images)
            
            if len(new_value) > 0:
                new_key = key[:-5] + '_mix4' + key[-5:]
                dest_image_address = os.path.join(destination_address, new_key)
                cv2.imwrite(dest_image_address, new_image)   
                augmentated_pickle.append([dest_image_address, new_value, 'mix4'])
            else:
                print("ignored")
                
        if augmentations['crop_mix4']:
            three_other_images = random.sample(data_pickle, 3)
            new_image, new_value = crop_mix4_image(image, value, three_other_images)
            
            if len(new_value) > 0:
                new_key = key[:-5] + '_crop_mix4' + key[-5:]
                dest_image_address = os.path.join(destination_address, new_key)
                cv2.imwrite(dest_image_address, new_image)   
                augmentated_pickle.append([dest_image_address, new_value, 'crop_mix4'])
            else:
                print("ignored")
            
    print("Length of data after augmentation is = " + str(len(augmentated_pickle)))
    pickle.dump(augmentated_pickle, open(augmentated_data_dict_address, 'wb'))
    
    return augmentated_pickle
    
    
if __name__ == '__main__':

    augmentated_pickle = create_more_data_using_augmentations('coco_train.pkl',
                                                              '/home/mohammad.hajizadeh/object_detection_datasets/CoCo2017/augment_train2017/',
                                                              'coco_train_augment.pkl')
                                          
    for augmentated_data in augmentated_pickle[:5]:
        print(augmentated_data[0], augmentated_data[2])
            
        
            
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                
                                            
