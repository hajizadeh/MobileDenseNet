import numpy as np
import cv2
import random


def create_default_prior_boxes(layer_width, layer_height, img_width, img_height, area, scales, aspect_ratios):
    box_widths = []
    box_heights = []
    for ar in aspect_ratios:
        for scale in scales:
            box_widths.append((area * scale) * np.sqrt(ar))
            box_heights.append((area * scale) / np.sqrt(ar))

    box_widths = np.array(box_widths)
    box_heights = np.array(box_heights)

    # define centers of prior boxes
    step_x = img_width / layer_width
    step_y = img_height / layer_height

    linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, int(layer_width))
    liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, int(layer_height))
    centers_x, centers_y = np.meshgrid(linx, liny)
    centers_x = centers_x.reshape(-1, 1)
    centers_y = centers_y.reshape(-1, 1)

    num_priors_ = len(aspect_ratios) * len(scales)
    prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
    prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))

    # This 4 line is for Absolute box codes
    # prior_boxes[:, ::4] -= box_widths / 2
    # prior_boxes[:, 1::4] -= box_heights / 2
    # prior_boxes[:, 2::4] += box_widths / 2
    # prior_boxes[:, 3::4] += box_heights / 2

    # This 2 line is for Relative box codes
    prior_boxes[:, 2::4] = box_widths
    prior_boxes[:, 3::4] = box_heights

    # This 2 line is for normalizing the boxes between 0, 1
    prior_boxes[:, ::2] /= img_width
    prior_boxes[:, 1::2] /= img_height

    prior_boxes = prior_boxes.reshape(-1, 4)

    # if clip:
    #     prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)

    # AFTER CLIP WE CAN CONVERT PRIORS TO RELATIVE AGAIN
    # prior_boxes_relative = np.zeros(np.shape(prior_boxes))
    # prior_boxes_relative[:, 0] = prior_boxes[:, 0] + ((prior_boxes[:, 2] - prior_boxes[:, 0]) / 2)
    # prior_boxes_relative[:, 1] = prior_boxes[:, 1] + ((prior_boxes[:, 3] - prior_boxes[:, 1]) / 2)
    # prior_boxes_relative[:, 2] = (prior_boxes[:, 2] - prior_boxes[:, 0])
    # prior_boxes_relative[:, 3] = (prior_boxes[:, 3] - prior_boxes[:, 1])
    # print("-------------------------------------------")

    return prior_boxes


def create_all_anchor_boxes(network_size, anchor_strides, anchor_areas, anchor_scales, anchor_aspect_ratios):
    prior_boxes_list = []

    for index in range(len(anchor_strides)):
        print("anchor box index = " + str(index))
        prior_boxes = create_default_prior_boxes(layer_width=np.ceil(network_size / anchor_strides[index]),
                                                 layer_height=np.ceil(network_size / anchor_strides[index]),
                                                 img_width=network_size,
                                                 img_height=network_size,
                                                 area=anchor_areas[index],
                                                 scales=anchor_scales[index],
                                                 aspect_ratios=anchor_aspect_ratios[index])
        print(np.shape(prior_boxes))
        prior_boxes_list.append(prior_boxes)

    all_prior_boxes = np.concatenate(prior_boxes_list, axis=0)
    print(np.shape(all_prior_boxes))
    return all_prior_boxes
    
    
    

