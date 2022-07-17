import tensorflow as tf
from utils import convert_to_corners


class BoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super(BoxLoss, self).__init__(
            reduction="none", name="BoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)
        
        
class IOULoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, anchor_boxes, box_variances):
        super(IOULoss, self).__init__(
            reduction="none", name="IOULoss"
        )
        self.box_variances = tf.convert_to_tensor(box_variances, dtype=tf.float32)
        self.anchor_boxes = anchor_boxes

    def call(self, y_true, y_pred):
        y_pred = y_pred * self.box_variances
        y_true = y_true * self.box_variances
        decode_y_true = tf.concat([self.anchor_boxes[:, 0:2] + (y_true[:, :, 0:2]) * self.anchor_boxes[:, 2:4],
                                   self.anchor_boxes[:, 2:4] * tf.math.exp(y_true[:, :, 2:4])], axis=-1)
        decode_y_pred = tf.concat([self.anchor_boxes[:, 0:2] + (y_pred[:, :, 0:2]) * self.anchor_boxes[:, 2:4],
                                   self.anchor_boxes[:, 2:4] * tf.math.exp(y_pred[:, :, 2:4])], axis=-1)
                                   
        decode_y_true = convert_to_corners(decode_y_true)
        decode_y_pred = convert_to_corners(decode_y_pred)
        
        inter_up_left = tf.maximum(decode_y_true[:, :, :2], decode_y_pred[:, :, :2])
        inter_bot_right = tf.minimum(decode_y_true[:, :, 2:4], decode_y_pred[:, :, 2:4])
        inter_wh = inter_bot_right - inter_up_left
        inter_wh = tf.maximum(inter_wh, 0)
        inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]

        # Compute union
        area_gt = (decode_y_true[:, :, 2] - decode_y_true[:, :, 0]) * (decode_y_true[:, :, 3] - decode_y_true[:, :, 1])
        area_pred = (decode_y_pred[:, :, 2] - decode_y_pred[:, :, 0]) * (decode_y_pred[:, :, 3] - decode_y_pred[:, :, 1])
        union = area_pred + area_gt - inter
        
        # Compute IoU
        iou = (inter / union)
        
        # Compute distance of centers        
        outer_up_left = tf.minimum(decode_y_true[:, :, :2], decode_y_pred[:, :, :2])
        outer_bot_right = tf.maximum(decode_y_true[:, :, 2:4], decode_y_pred[:, :, 2:4])
        
        outer_wh = outer_bot_right - outer_up_left
        outer_wh = tf.maximum(outer_wh, 1e-8)
        outer_diagonal_pow_2 = tf.math.square(outer_wh[:, :, 0]) + tf.math.square(outer_wh[:, :, 1])
        
        center_diff = tf.math.square(0.5 * (decode_y_true[:, :, 2:4] + decode_y_true[:, :, 0:2]) - 0.5 * (decode_y_pred[:, :, 2:4] + decode_y_pred[:, :, 0:2]))
        center_diff = tf.reduce_sum(center_diff, axis=-1) / outer_diagonal_pow_2
                                    
        loss = (1. - iou) + center_diff        
        
        return loss


class ClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super(ClassificationLoss, self).__init__(
            reduction="none", name="ClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        # FOCAL LOSS SHIT
        # probs = tf.nn.sigmoid(y_pred)
        # alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        # pt = tf.where(tf.equal(y_true, 1.), 1. - probs, probs)
        # # pt = tf.reduce_sum(pt, axis=-1)
        # loss = alpha * tf.pow(pt, self._gamma) * cross_entropy
        # return tf.reduce_sum(loss, axis=-1)
        return cross_entropy
        

class Loss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_classes, anchor_boxes, box_variances, alpha=0.25, gamma=1., delta=1., hnm=True, neg_pos_ratio=3., negatives_for_hard=3.):
        super(Loss, self).__init__(reduction="none", name="Loss")
        self._clf_loss = ClassificationLoss(alpha, gamma)
        self._box_loss = BoxLoss(delta)
        # TODO: Test IOU Loss
        # self._box_loss = IOULoss(anchor_boxes, box_variances)
        self._num_classes = num_classes
        self.hnm = hnm
        self.neg_pos_ratio = neg_pos_ratio
        self.negatives_for_hard = negatives_for_hard

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        num_boxes = tf.shape(y_true)[1]
        batch_size = tf.shape(y_true)[0]

        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        negative_mask = tf.cast(tf.equal(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)

        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]

        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_labels = tf.concat([cls_labels, tf.expand_dims(negative_mask, -1)], axis=-1)
        
        cls_predictions = y_pred[:, :, 4:]

        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)

        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        pos_box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        pos_clf_loss = tf.where(tf.equal(positive_mask, 1.0), clf_loss, 0.0)
        neg_clf_loss = tf.where(tf.equal(negative_mask, 1.0), clf_loss, 0.0)

        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        negative_nums = tf.minimum(normalizer * tf.cast(self.neg_pos_ratio, dtype=tf.float32),
                                   tf.cast(num_boxes, dtype=tf.float32) - tf.cast(normalizer, dtype=tf.float32))
        negative_nums = tf.where(tf.equal(negative_nums, 0.0), tf.cast(self.negatives_for_hard, dtype=tf.float32), negative_nums)
        normalizer = tf.where(tf.equal(normalizer, 0.0), tf.cast(self.negatives_for_hard, dtype=tf.float32), normalizer)
        
        if self.hnm:
            rank = tf.argsort(neg_clf_loss, axis=1, direction='DESCENDING')
            rank = tf.argsort(rank, axis=1)
            final_negative_mask = tf.cast(tf.cast(rank, dtype=tf.float32) < tf.expand_dims(negative_nums, 1), dtype=tf.float32)        
            neg_clf_loss = tf.where(tf.equal(final_negative_mask, 1.0), neg_clf_loss, 0.0)
        
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        pos_clf_loss = tf.math.divide_no_nan(tf.reduce_sum(pos_clf_loss, axis=-1), normalizer)  # + negative_nums_batches
        pos_box_loss = tf.math.divide_no_nan(tf.reduce_sum(pos_box_loss, axis=-1), normalizer)
        neg_clf_loss = tf.math.divide_no_nan(tf.reduce_sum(neg_clf_loss, axis=-1), normalizer)

        loss = pos_clf_loss + neg_clf_loss + pos_box_loss
        return pos_box_loss, pos_clf_loss, neg_clf_loss, loss


class AccuracyF1(object):

    def __init__(self, num_class, anchor_boxes, box_variances):
        self.num_class = num_class
        self.box_variances = tf.convert_to_tensor(box_variances, dtype=tf.float32)
        self.anchor_boxes = anchor_boxes

    def classification_accuracy(self, y_true, y_pred):
        classification_accuracy = tf.reduce_sum(y_true * y_pred, axis=-1)
        classification_accuracy = tf.where(tf.math.greater(classification_accuracy, 0.5),
                                           tf.constant(1.0, dtype=tf.float32),
                                           tf.constant(0.0, dtype=tf.float32))

        return classification_accuracy

    def localization_accuracy(self, y_true, y_pred, threshold=0.7):
        localization_difference = y_true - y_pred
        localization_mask = tf.reduce_sum(tf.where(tf.math.less(localization_difference, threshold), 1.0, 0.0), axis=-1)
        localization_acc = tf.where(tf.math.equal(localization_mask, 4.0), 1.0, 0.0)

        return localization_acc

    def real_iou_accuracy(self, y_true, y_pred, iou_threshold=0.5):
        # First you need to decode ground truth boxes and predictions
        y_pred = y_pred * self.box_variances
        y_true = y_true * self.box_variances
        decode_y_true = tf.concat([self.anchor_boxes[:, 0:2] + (y_true[:, :, 0:2]) * self.anchor_boxes[:, 2:4],
                                   self.anchor_boxes[:, 2:4] * tf.math.exp(y_true[:, :, 2:4])], axis=-1)
        decode_y_pred = tf.concat([self.anchor_boxes[:, 0:2] + (y_pred[:, :, 0:2]) * self.anchor_boxes[:, 2:4],
                                   self.anchor_boxes[:, 2:4] * tf.math.exp(y_pred[:, :, 2:4])], axis=-1)

        inter_up_left = tf.maximum((decode_y_true[:, :, :2] - (decode_y_true[:, :, 2:] / 2)),
                                   (decode_y_pred[:, :, :2] - decode_y_pred[:, :, 2:] / 2))
        inter_bot_right = tf.minimum((decode_y_true[:, :, :2] + (decode_y_true[:, :, 2:] / 2)),
                                     (decode_y_pred[:, :, :2] + decode_y_pred[:, :, 2:] / 2))
        inter_wh = inter_bot_right - inter_up_left
        inter_wh = tf.maximum(inter_wh, 0)
        inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]

        # Compute union
        area_pred = (decode_y_pred[:, :, 2]) * (decode_y_pred[:, :, 3])
        area_gt = (decode_y_true[:, :, 2]) * (decode_y_true[:, :, 3])
        union = area_pred + area_gt - inter

        # Compute iou
        iou = (inter / union)

        iou_accuracy = tf.where(tf.math.greater(iou, iou_threshold),
                                tf.constant(1.0, dtype=tf.float32),
                                tf.constant(0.0, dtype=tf.float32))

        return iou_accuracy

    def compute_accuracy(self, y_true, y_pred):
        # active softmax here on y_pred
        y_pred_softmax = tf.nn.softmax(y_pred[:, :, 4:])[:, :, :-1]
        # y_pred_softmax = tf.nn.sigmoid(y_pred[:, :, 4:])

        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)

        y_true_labels = tf.one_hot(tf.cast(y_true[:, :, 4], dtype=tf.int32), self.num_class, dtype=tf.float32)

        localization_accuracy = self.real_iou_accuracy(y_true[:, :, 0:4], y_pred[:, :, 0:4])
        localization_accuracy_75 = self.real_iou_accuracy(y_true[:, :, 0:4], y_pred[:, :, 0:4], iou_threshold=0.75)
        # WHEN YOU HAVE DYNAMIC ANCHORS
        # localization_accuracy = self.localization_accuracy(y_true[:, :, 0:4], y_pred[:, :, 0:4])
        classification_accuracy = self.classification_accuracy(y_true_labels, y_pred_softmax)

        positive_nums = tf.reduce_sum(positive_mask, axis=-1)
        positive_nums = tf.where(tf.equal(positive_nums, 0.0),
                                 tf.constant(1.0, dtype=tf.float32),
                                 tf.cast(positive_nums, dtype=tf.float32))

        y_pred_positive = tf.reduce_sum(tf.where(tf.math.greater(y_pred_softmax, 0.5),
                                                 tf.constant(1.0, dtype=tf.float32),
                                                 tf.constant(0.0, dtype=tf.float32)), axis=-1)

        positive_nums_predictions = tf.reduce_sum(y_pred_positive, axis=-1)
        positive_nums_predictions = tf.where(tf.equal(positive_nums_predictions, 0.0),
                                             tf.constant(1.0, dtype=tf.float32),
                                             tf.cast(positive_nums_predictions, dtype=tf.float32))

        accuracy_combined = localization_accuracy * classification_accuracy
        accuracy_combined_75 = localization_accuracy_75 * classification_accuracy

        recall_class = tf.reduce_sum(classification_accuracy, axis=-1) / positive_nums
        recall = tf.reduce_sum(accuracy_combined, axis=-1) / positive_nums
        recall_75 = tf.reduce_sum(accuracy_combined_75, axis=-1) / positive_nums
        precision_class = tf.reduce_sum(classification_accuracy, axis=-1) / positive_nums_predictions
        precision = tf.reduce_sum(accuracy_combined, axis=-1) / positive_nums_predictions
        precision_75 = tf.reduce_sum(accuracy_combined_75, axis=-1) / positive_nums_predictions

        precision_plus_recall_class = precision_class + recall_class
        precision_plus_recall = precision + recall
        precision_plus_recall_75 = precision_75 + recall_75
        
        precision_plus_recall_class = tf.where(tf.equal(precision_plus_recall_class, 0.0),
                                               tf.constant(1.0, dtype=tf.float32),
                                               precision_plus_recall_class)
        precision_plus_recall = tf.where(tf.equal(precision_plus_recall, 0.0),
                                         tf.constant(1.0, dtype=tf.float32),
                                         precision_plus_recall)
        precision_plus_recall_75 = tf.where(tf.equal(precision_plus_recall_75, 0.0),
                                          tf.constant(1.0, dtype=tf.float32),
                                          precision_plus_recall_75)

        total_f1_class = (2 * recall_class * precision_class) / precision_plus_recall_class
        total_f1 = (2 * recall * precision) / precision_plus_recall
        total_f1_75 = (2 * recall_75 * precision_75) / precision_plus_recall_75

        return recall, precision, total_f1, total_f1_75


