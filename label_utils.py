import tensorflow as tf
import numpy as np
from utils import convert_to_corners, random_flip_horizontal, my_iou, convert_to_xywh, compute_iou, preprocess_single_image
from utils import random_padding_low, random_zoom_in_low, adjust_brightness_and_contrast_low, adjust_hue_and_saturation_low
from utils import random_padding, random_zoom_in, adjust_brightness_and_contrast, adjust_hue_and_saturation


class LabelUtils:
    def __init__(self, network_size, num_classes, match_iou, ignore_iou, box_variances, anchor_boxes,
                confidence_threshold, nms_iou_threshold, max_predictions):
        self.network_size = network_size
        self.num_classes = num_classes
        self.match_iou = match_iou
        self.ignore_iou = ignore_iou
        self.box_variances = tf.convert_to_tensor(box_variances, dtype=tf.float32)
        self.anchor_boxes = anchor_boxes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_predictions = max_predictions
                                                                        
    def compute_box_target(self, matched_gt_boxes):
        """Transforms and decode the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - self.anchor_boxes[:, :2]) / self.anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / self.anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self.box_variances
        return box_target
        
    def decode_box_predictions(self, box_predictions):
        boxes = box_predictions * self.box_variances
        boxes = tf.concat(
            [
                boxes[:, :2] * self.anchor_boxes[:, 2:] + self.anchor_boxes[:, :2],
                tf.math.exp(boxes[:, 2:]) * self.anchor_boxes[:, 2:],
            ],
            axis=-1,
        )
        return boxes
        
    def preprocess_data(self, file_address, labels):
        """Applies pre processing step to a single sample
    
        Arguments:
          file_address: An image address.
          labels: Image labels and bbox.
    
        Returns:
          resized_image: Resized and padded image with random data augmentation applied.
          decodel_labels: Decoded ground truth boxes and classes with the shape `(num_anchors, 5)`
        """
        
        labels = labels.to_tensor()
    
        image = tf.io.read_file(file_address)
        image = tf.io.decode_jpeg(image, channels=3)
        print(image.dtype)
    
        bbox = convert_to_corners(labels[:, 0:4])
        class_id = tf.cast(labels[:, 4], dtype=tf.float32)
    
        image, bbox = random_flip_horizontal(image, bbox)
        
        image = adjust_brightness_and_contrast(image)
        image = adjust_hue_and_saturation(image)
        print(image.dtype)
        
        if tf.random.uniform(()) > 0.3:
            image, bbox = random_padding_low(image, bbox)
        print(image.dtype)
    
        image, bbox = random_zoom_in(image, bbox)
        print(image.dtype)
    
        resize_image = tf.image.resize(image, (self.network_size, self.network_size))
        print(resize_image.dtype)
    
        bbox = convert_to_xywh(bbox)
    
        # MAYBE GATHER POSITIVE BOXES
        bbox = tf.maximum(bbox, 0.001)
    
        iou_matrix = compute_iou(self.anchor_boxes, bbox)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, self.match_iou)
        negative_mask = tf.less(max_iou, self.ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
    
        positive_mask = tf.cast(positive_mask, dtype=tf.float32)
        ignore_mask = tf.cast(ignore_mask, dtype=tf.float32)
    
        matched_gt_boxes = tf.gather(bbox, matched_gt_idx)
        box_target = self.compute_box_target(matched_gt_boxes)
    
        matched_gt_cls_ids = tf.gather(class_id, matched_gt_idx)
        cls_target = tf.where(tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids)
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        decoded_label = tf.concat([box_target, cls_target], axis=-1)
    
        resize_image = preprocess_single_image(resize_image)
    
        print(resize_image.shape)
        print(decoded_label.dtype)
        print(decoded_label.shape)
    
        return resize_image, decoded_label
        
    def preprocess_data_validation(self, file_address, labels):
        """Applies pre processing step to a single sample
    
        Arguments:
          file_address: An image address.
          labels: Image labels and bbox.
    
        Returns:
          resized_image: Resized and padded image with random data augmentation applied.
          decodel_labels: Decoded ground truth boxes and classes with the shape `(num_anchors, 5)`
        """
        labels = labels.to_tensor()
    
        image = tf.io.read_file(file_address)
        image = tf.io.decode_jpeg(image, channels=3)
    
        bbox = convert_to_corners(labels[:, 0:4])
        class_id = tf.cast(labels[:, 4], dtype=tf.float32)
    
        resize_image = tf.image.resize(image, (self.network_size, self.network_size))
    
        bbox = convert_to_xywh(bbox)
        
        # MAYBE GATHER POSITIVE BOXES
        # bbox = tf.maximum(bbox, 1e-3)
    
        iou_matrix = compute_iou(self.anchor_boxes, bbox)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, self.match_iou)
        negative_mask = tf.less(max_iou, self.ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
    
        positive_mask = tf.cast(positive_mask, dtype=tf.float32)
        ignore_mask = tf.cast(ignore_mask, dtype=tf.float32)
    
        matched_gt_boxes = tf.gather(bbox, matched_gt_idx)
        box_target = self.compute_box_target(matched_gt_boxes)
    
        matched_gt_cls_ids = tf.gather(class_id, matched_gt_idx)
        cls_target = tf.where(tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids)
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        decoded_label = tf.concat([box_target, cls_target], axis=-1)
    
        resize_image = preprocess_single_image(resize_image)
    
        return resize_image, decoded_label
        
    def non_maximum_suppression(self, predictions_to_process, threshold):
    
        if len(predictions_to_process) == 1:
            return np.array(predictions_to_process)
            
        best_predictions = []
        argsort = np.argsort(predictions_to_process[:, 1])
        sorted_predictions = predictions_to_process[argsort[::-1]]
    
        while len(sorted_predictions) > 0:
            prediction = sorted_predictions[0]
            best_predictions.append(prediction)
    
            sorted_predictions = np.delete(sorted_predictions, 0, axis=0)
            ious = my_iou(sorted_predictions[:, 2:], prediction[2:])
    
            departed_indexes = ious > threshold
            if len(sorted_predictions[departed_indexes]) > 0:
                indices = []
                for index, bool in enumerate(departed_indexes):
                    if bool:
                        indices.append(index)
    
                sorted_predictions = np.delete(sorted_predictions, indices, axis=0)
    
        best_predictions = np.array(best_predictions)        
        return best_predictions
        
    def non_maximum_suppression_v2(self, predictions_to_process, threshold):
            
        best_predictions = []
        argsort = np.argsort(predictions_to_process[:, 1])
        sorted_predictions = predictions_to_process[argsort[::-1]]
        # print(tf.argsort(predictions_to_process[:, 1], direction='DESCENDING'))
        # sorted_predictions = tf.gather(predictions_to_process, tf.argsort(predictions_to_process[:, 1], direction='DESCENDING'))
    
        while sorted_predictions.shape[0] > 0:
            prediction = sorted_predictions[0]
            best_predictions.append(prediction)
            
            ious = my_iou(sorted_predictions[1:, 2:], prediction[2:])
            keep_indexes = ious <= threshold                
            sorted_predictions = sorted_predictions[1:][keep_indexes]
    
        # best_predictions = tf.stack(best_predictions) 
        return best_predictions
        
    def extract_detections_from_predictions_old(self, predictions):
    
        # (batch_size, 8096, 25)
        locations_offsets = predictions[:, :, :4]
        
        # SSD
        prediction_confidences = tf.nn.softmax(predictions[:, :, 4:])  # [:, :, :-1]
        
        # RETINA
        # prediction_confidences = tf.nn.sigmoid(predictions[:, :, 4:])
        
        # YOLO
        # obj_sigmoid = tf.nn.sigmoid(predictions[:, :, -1])
        # prediction_mask = tf.expand_dims(tf.where(tf.greater(obj_sigmoid, 0.1), obj_sigmoid, 0.0), axis=2)
        # prediction_confidences = tf.nn.softmax(predictions[:, :, 4:-1]) * prediction_mask
        # prediction_confidences = obj_sigmoid
    
        detections = []
        for batch_index in range(len(predictions)):
            detections.append([])
            decoded_boxes = self.decode_box_predictions(locations_offsets[batch_index])
    
            for class_index in range(self.num_classes):
                confidences = prediction_confidences[batch_index, :, class_index]
                good_confidences = confidences > self.confidence_threshold
    
                if len(confidences[good_confidences]) > 0:
                    boxes_to_process = decoded_boxes[good_confidences]
                    confidences_to_process = confidences[good_confidences]
                    
                    labels = class_index * np.ones((len(confidences_to_process)))    
                    predictions_to_process = np.zeros((len(confidences_to_process), 6))
                    predictions_to_process[:, 0] = labels
                    predictions_to_process[:, 1] = confidences_to_process
                    predictions_to_process[:, 2:] = boxes_to_process
    
                    best_predictions = self.non_maximum_suppression_v2(predictions_to_process, self.nms_iou_threshold)
                    # best_predictions = tf.gather(predictions_to_process, tf.image.non_max_suppression(convert_to_corners(boxes_to_process), confidences_to_process,
                    #                                                                                   self.max_predictions, iou_threshold=self.nms_iou_threshold))
    
                    detections[batch_index].extend(best_predictions)
                    
            if len(detections[batch_index]) > 0:
                detections_batch = np.array(detections[batch_index])
                # detections_batch = non_maximum_suppression(detections_batch, threshold=0.6)
                argsort = np.argsort(detections_batch[:, 1]) 
                detections[batch_index] = detections_batch[argsort[::-1]][:self.max_predictions]  # -1:-max_predictions -1:-1
                
            
            if len(detections[batch_index]) > self.max_predictions * (2./3.):
                print(len(detections[batch_index]))
                print('----------------------')
                
        return detections
        
    def extract_detections_from_predictions(self, predictions):
        # NOT TESTED YET
    
        locations_offsets = predictions[:, :, :4]
        prediction_confidences = tf.nn.softmax(predictions[:, :, 4:])  # [:, :, :-1]
    
        detections = []
        for batch_index in range(len(predictions)):
            detections.append([])
            decoded_boxes = self.decode_box_predictions(locations_offsets[batch_index])
    
            for class_index in range(self.num_classes):
                confidences = prediction_confidences[batch_index, :, class_index]
                # good_confidences = confidences > self.confidence_threshold
                # labels = class_index * np.ones((len(confidences)))

                best_indices = tf.image.non_max_suppression(convert_to_corners(decoded_boxes), confidences, max_output_size=self.max_predictions,
                                                            iou_threshold=self.nms_iou_threshold, score_threshold=self.confidence_threshold)
                
                best_predictions = tf.concat([tf.expand_dims(class_index * tf.ones((len(best_indices))), axis=-1),
                                            tf.expand_dims(tf.gather(confidences, best_indices), axis=-1),
                                            tf.gather(decoded_boxes, best_indices)], axis=-1)                
    
                detections[batch_index].extend(best_predictions.numpy())
                    
            if len(detections[batch_index]) > 0:
                detections_batch = np.array(detections[batch_index])
                # detections_batch = non_maximum_suppression(detections_batch, threshold=0.6)
                argsort = np.argsort(detections_batch[:, 1]) 
                detections[batch_index] = detections_batch[argsort[::-1]][:self.max_predictions]  # -1:-max_predictions -1:-1              
            
            if len(detections[batch_index]) > self.max_predictions * (2./3.):
                print(len(detections[batch_index]))
                print('----------------------')
                
        return detections
        
    def extract_detections_from_predictions_argmax(self, predictions):
        # NOT TESTED YET
    
        locations_offsets = predictions[:, :, :4]
        prediction_confidences = tf.nn.softmax(predictions[:, :, 4:])[:, :, :-1]
        # prediction_classes = tf.nn.argmax(prediction_confidences, axis=-1)
    
        detections = []
        counter = []
        for batch_index in range(len(predictions)):
            detections.append([])
            counter.append([])
            decoded_boxes = self.decode_box_predictions(locations_offsets[batch_index])
            confidences = prediction_confidences[batch_index]
            classes_indices = tf.cast(tf.math.argmax(confidences, axis=-1), dtype=tf.float32)
            classes_confidences = tf.math.reduce_max(confidences, axis=-1)

            best_indices = tf.image.non_max_suppression(convert_to_corners(decoded_boxes), classes_confidences, max_output_size=self.max_predictions,
                                                        iou_threshold=self.nms_iou_threshold, score_threshold=self.confidence_threshold)
            
            best_predictions = tf.concat([tf.expand_dims(tf.gather(classes_indices, best_indices), axis=-1),
                                          tf.expand_dims(tf.gather(classes_confidences, best_indices), axis=-1),
                                          tf.gather(decoded_boxes, best_indices)], axis=-1)
                                          
            # print(type(best_predictions))
            detections[batch_index].extend(best_predictions.numpy())
            counter[batch_index] = len(detections[batch_index])
                    
            # if len(detections[batch_index]) > 0:
                # detections_batch = non_maximum_suppression(detections_batch, threshold=0.6)
            
            # if len(detections[batch_index]) > self.max_predictions * (2./3.):
            #     print(len(detections[batch_index]))
            #     print('----------------------')
                
        # print(type(detections))
                
        return detections, counter








