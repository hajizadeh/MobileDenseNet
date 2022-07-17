import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans



data_dict_address = "imagenet_vid_train.pkl"
validation_dict_address = 'imagenet_vid_val.pkl'

train_data_information = pickle.load(open(data_dict_address, "rb"))
validation_data_information = pickle.load(open(validation_dict_address, "rb"))

all_data_information = train_data_information + validation_data_information

image_size = 320
min_ignore_area = 12.
min_ignore_ratio = 1. / 5.
max_ignore_ratio = 5.0
ratio_per_area = [3, 5, 5]
scales_per_featuremap = 3
areas = []
for address, label, augment_type in all_data_information:
    for bbox in label:
        if bbox[2] == 0. or bbox[3] == 0.:
            print(address)
            print(bbox)
        ratio = (bbox[2] * image_size) / (bbox[3] * image_size)
        area = np.sqrt((bbox[2] * image_size) * (bbox[3] * image_size))
        if area < min_ignore_area or ratio < min_ignore_ratio or ratio > max_ignore_ratio:
            continue
        areas.append([area])

areas_array = np.array(areas)

kmeans = KMeans(n_clusters=len(ratio_per_area) * scales_per_featuremap, n_init=30, max_iter=2000, random_state=20).fit(areas_array)
areas_list = list(np.sort(kmeans.cluster_centers_[:, 0]))
print("----------------------")
print(areas_list)
print(areas_list[::scales_per_featuremap])
print("----------------------")

scales = []
for idx, area in enumerate(areas_list):
    if idx % scales_per_featuremap == 0:
        if scales_per_featuremap == 1:
            scales.append([1.])
        if scales_per_featuremap == 2:
            scales.append([1., round(areas_list[idx + 1] / areas_list[idx], 3)])
        if scales_per_featuremap == 3:
            scales.append([1., round(areas_list[idx + 1] / areas_list[idx], 3), round(areas_list[idx + 2] / areas_list[idx], 3)])
        
print(scales)
print("----------------------")

area_thresholds = []
for idx, area in enumerate(areas_list):
    if idx % scales_per_featuremap == 0 and idx > 0:
        area_thresholds.append((areas_list[idx - 1] + areas_list[idx]) / 2)
        
print(area_thresholds)
print("----------------------")

ratios = []
for i in range(len(ratio_per_area)):
    ratios.append([])
    
for address, label, augment_type in all_data_information:
    for bbox in label:
        ratio = (bbox[2] * image_size) / (bbox[3] * image_size)
        area = np.sqrt((bbox[2] * image_size) * (bbox[3] * image_size))
        if area < min_ignore_area or ratio < min_ignore_ratio or ratio > max_ignore_ratio:
            continue
            
        for index, at in enumerate(area_thresholds):
            if area < at:
                ratios[index].append([((bbox[2] * image_size) / (bbox[3] * image_size))])
        if area > area_thresholds[-1]:
            ratios[-1].append([((bbox[2] * image_size) / (bbox[3] * image_size))])

# print(np.shape(ratios))
aspect_ratios = []
for idx, ratio in enumerate(ratios):
    ratio = np.array(ratio)
    print(np.mean(ratio), np.max(ratio), np.min(ratio))
    print(ratio.shape)
    kmeans = KMeans(n_clusters=ratio_per_area[idx], n_init=30, max_iter=2000, random_state=20).fit(ratio)
    aspect_ratio = list(np.sort(kmeans.cluster_centers_[:, 0]))
    aspect_ratios.append(list(np.round(aspect_ratio, 3)))

print("----------------------")
print(aspect_ratios)
print("----------------------")

print("++++++++++++++++++++++++")
print(list(np.round(areas_list[::scales_per_featuremap], 3)))
print(scales)
print(aspect_ratios)


# cluster_centers_areas = np.array(kmeans.cluster_centers_[:, 0]) * np.array(kmeans.cluster_centers_[:, 1])

# cluster_widths = kmeans.cluster_centers_[np.argsort(cluster_centers_areas), 0]
# cluster_heights = kmeans.cluster_centers_[np.argsort(cluster_centers_areas), 1]

# print(cluster_widths)
# print(cluster_heights)
    
# plt.scatter(cluster_widths, cluster_heights)
# plt.savefig('kmeans.png')

# plt.hist(all_widths)
# plt.savefig('all_widths.png')

# plt.hist(all_heights)
# plt.savefig('all_heights.png')

# print(object_names)
    
