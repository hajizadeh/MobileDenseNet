import tensorflow as tf
import numpy as np


def preprocess_single_image(image):
    # image = image / 255.
    # image = tf.keras.applications.resnet.preprocess_input(image)
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    # image = image / 127.5 - 1.
    return image
    

def convert_to_xywh(boxes):
    # Changes the box format to center, width and height.
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    # Changes the box format to corner coordinates        
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
      boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)
    
    
def my_iou(all_boxes, box):
    # Compute intersection over union for the box with all priors.
    # Compute intersection
    inter_up_left = np.maximum(all_boxes[:, :2] - (all_boxes[:, 2:]/2), box[:2] - (box[2:]/2))
    inter_bot_right = np.minimum(all_boxes[:, :2] + (all_boxes[:, 2:]/2), box[:2] + (box[2:]/2))
    inter_wh = inter_bot_right - inter_up_left
    inter_wh = np.maximum(inter_wh, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]

    # Compute union
    area_gt = box[2] * box[3]
    area_pred = (all_boxes[:, 2]) * (all_boxes[:, 3])
    union = area_pred + area_gt - inter

    # Compute iou
    iou = inter / union
    return iou


def adjust_brightness_and_contrast(image):
    # Adjust brightness and contrast of an image with 50% chance
    if tf.random.uniform(()) > 0.3:
        image = tf.image.random_brightness(image, 0.1)

    if tf.random.uniform(()) > 0.3:
        image = tf.image.random_contrast(image, 0.8, 1.3)

    return image
    
    
def adjust_brightness_and_contrast_low(image):
    # Adjust brightness and contrast of an image with 50% chance
    brightness_rand = tf.random.uniform(())
    if brightness_rand > 0.75:
        image = tf.image.adjust_brightness(image, 0.1)
    elif brightness_rand > 0.5:
        image = tf.image.adjust_brightness(image, -0.1)
    
    constrast_rand = tf.random.uniform(())
    if constrast_rand > 0.75:
        image = tf.image.adjust_contrast(image, 1.2)
    elif constrast_rand > 0.5:
        image = tf.image.adjust_contrast(image, 0.85)

    return image


def adjust_hue_and_saturation(image):
    # Adjust hue and saturation of an image with 50% chance
    if tf.random.uniform(()) > 0.3:
        image = tf.image.random_hue(image, 0.1)

    if tf.random.uniform(()) > 0.3:
        image = tf.image.random_saturation(image, 0.8, 1.25)

    return image
    
    
def adjust_hue_and_saturation_low(image):
    # Adjust hue and saturation of an image with 50% chance
    hue_rand = tf.random.uniform(())
    if hue_rand > 0.75:
        image = tf.image.adjust_hue(image, 0.1)
    elif hue_rand > 0.5:
        image = tf.image.adjust_hue(image, -0.1)

    saturation_rand = tf.random.uniform(())
    if saturation_rand > 0.75:
        image = tf.image.adjust_saturation(image, 1.15)
    elif saturation_rand > 0.5:
        image = tf.image.adjust_saturation(image, 0.9)

    return image


def random_flip_horizontal(image, boxes):
    # Flips image and boxes horizontally with 50% chance
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes
    
    
def random_padding(image, bbox, zero_stride=64.):
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)    
    random_pad_size = tf.math.floor(tf.random.uniform((2,), 0., zero_stride, dtype=tf.float32))
    padded_image_shape = image_shape + random_pad_size
    offsets = tf.math.floor(tf.random.uniform((2,), 0., random_pad_size, dtype=tf.float32))
    
    paddings = tf.convert_to_tensor([[tf.cast(offsets[0], dtype=tf.int32),
                                    tf.cast(random_pad_size[0] - offsets[0], dtype=tf.int32)],
                                    [tf.cast(offsets[1], dtype=tf.int32),
                                    tf.cast(random_pad_size[1] - offsets[1], dtype=tf.int32)],
                                    [0, 0]])
    
    image = tf.pad(image, paddings, mode="REFLECT")  # CONSTANT, REFLECT, SYMMETRIC
    
    offsets = offsets / image_shape
    ratio_after = image_shape / padded_image_shape
    bbox = tf.stack(
        [
            (bbox[:, 0] + offsets[1]) * ratio_after[1],
            (bbox[:, 1] + offsets[0]) * ratio_after[0],
            (bbox[:, 2] + offsets[1]) * ratio_after[1],
            (bbox[:, 3] + offsets[0]) * ratio_after[0],
        ],
        axis=-1,
    )

    return image, bbox

    
def random_padding_low(image, bbox, zero_stride=64.):
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)    
    random_pad_size = zero_stride / tf.math.pow(2., tf.math.floor(tf.random.uniform((), 1., 3., dtype=tf.float32)))
    padded_image_shape = image_shape + random_pad_size
    offsets = tf.math.floor(tf.random.uniform((2,), 0., random_pad_size, dtype=tf.float32))
    
    paddings = tf.convert_to_tensor([[tf.cast(offsets[0], dtype=tf.int32),
                                    tf.cast(random_pad_size - offsets[0], dtype=tf.int32)],
                                    [tf.cast(offsets[1], dtype=tf.int32),
                                    tf.cast(random_pad_size - offsets[1], dtype=tf.int32)],
                                    [0, 0]])
    
    image = tf.pad(image, paddings, mode="REFLECT")  # CONSTANT, REFLECT, SYMMETRIC
    
    offsets = offsets / image_shape
    ratio_after = image_shape / padded_image_shape
    bbox = tf.stack(
        [
            (bbox[:, 0] + offsets[1]) * ratio_after[1],
            (bbox[:, 1] + offsets[0]) * ratio_after[0],
            (bbox[:, 2] + offsets[1]) * ratio_after[1],
            (bbox[:, 3] + offsets[0]) * ratio_after[0],
        ],
        axis=-1,
    )

    return image, bbox
    
    
def random_zoom_in(image, bbox):    
    crop_top_left_rand = tf.random.uniform(())
    crop_bot_right_rand = tf.random.uniform(())

    if crop_top_left_rand < 0.8:
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        crop_top_left_rand = crop_top_left_rand / 3.
        image = image[int(image_shape[0] * crop_top_left_rand):, int(image_shape[1] * crop_top_left_rand):, :]
        bbox = tf.stack(
            [
                tf.clip_by_value((bbox[:, 0] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 1] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 2] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 3] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0)
            ],
            axis=-1,
        )

    if crop_bot_right_rand < 0.8:
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        crop_bot_right_rand = crop_bot_right_rand / 3.
        image = image[:int(image_shape[0] * (1. - crop_bot_right_rand)), :int(image_shape[1] * (1. - crop_bot_right_rand)), :]
        bbox = tf.stack(
            [
                tf.clip_by_value((bbox[:, 0]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 1]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 2]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 3]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0)
            ],
            axis=-1,
        )

    return image, bbox
    
    
def random_zoom_in_low(image, bbox):    
    crop_top_left_rand = tf.math.floor(tf.random.uniform((), 1., 7., dtype=tf.float32)) / 10.        
    crop_bot_right_rand = tf.math.floor(tf.random.uniform((), 1., 7., dtype=tf.float32)) / 10.

    if crop_top_left_rand < 0.35:
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        image = image[int(image_shape[0] * crop_top_left_rand):, int(image_shape[1] * crop_top_left_rand):, :]
        bbox = tf.stack(
            [
                tf.clip_by_value((bbox[:, 0] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 1] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 2] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 3] - crop_top_left_rand) * (1 / (1 - crop_top_left_rand)), 0.0, 1.0)
            ],
            axis=-1,
        )

    if crop_bot_right_rand < 0.35:
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        image = image[:int(image_shape[0] * (1. - crop_bot_right_rand)), :int(image_shape[1] * (1. - crop_bot_right_rand)), :]
        bbox = tf.stack(
            [
                tf.clip_by_value((bbox[:, 0]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 1]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 2]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0),
                tf.clip_by_value((bbox[:, 3]) * (1 / (1 - crop_bot_right_rand)), 0.0, 1.0)
            ],
            axis=-1,
        )

    return image, bbox











