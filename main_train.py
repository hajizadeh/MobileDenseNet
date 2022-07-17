import os
import pickle
import random
import timeit
import math
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from config import network_size, num_classes, match_iou, ignore_iou, confidence_threshold, nms_iou_threshold, max_predictions
from config import anchor_strides, anchor_areas, anchor_scales, anchor_aspect_ratios, anchor_per_grid, box_variances
from create_default_boxes import create_all_anchor_boxes

from loss_accuracy import Loss, AccuracyF1
from label_utils import LabelUtils

from mobilenet_ssdlite import create_mobilenet_ssdlite


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ARGPARSE (EVERYTHING THAT YOU NEED TO CHANGE)
# DEFINE ALL DATASET ARGUMENTS HERE
train_dict_address = "coco_train_augment.pkl"
validation_dict_address = "coco_val.pkl"

# DEFINE ALL TRAIN ARGUMENTS HERE
checkpoints_folder = "checkpoints_mobilenet_best2"
trainable_backbone = True
resume = False
resume_checkpoint = "checkpoints_mobilenet_freeze2/model_weights_best_loss_0004.h5"

epochs = 240
warmup_epochs = 0.5
train_epoch_steps = 1000
start_validating = -1
batch_size = 64

learning_rate = 0.1
momentum = 0.9
weight_decay = 0.0003

# LRS SHIT
lr_scheduler = 'cosine'  # cosine or piecewise or rop
learning_rates = [learning_rate * 0.1, learning_rate, learning_rate * 0.1, learning_rate * 0.01, learning_rate * 0.001]
learning_rate_boundaries = [300, 70 * train_epoch_steps, 100 * train_epoch_steps, 120 * train_epoch_steps]
t_max = epochs - 5
cosine_alpha = 0.001
CHANGE_LR_BASED_ON_PEALATEU = False
reduce_on_plt_patients = 8
learning_rate_factor = 0.1

# CREATE ANCHOR BOXES
anchor_boxes = tf.convert_to_tensor(create_all_anchor_boxes(network_size, anchor_strides, anchor_areas,
                                                            anchor_scales, anchor_aspect_ratios), dtype=tf.float32)

# TRAIN THE MODEL WITH CUSTOM LOOP
validation_history = {'pos_loss': [], 'neg_loss': [], 'loc_loss': [], 'loss': [],
                      'recall': [], 'precision': [], 'F1': [], 'F1_75': [],
                      'val_pos_loss': [], 'val_neg_loss': [], 'val_class_loss': [], 'val_loc_loss': [], 'val_loss': [],
                      'val_recall': [], 'val_precision': [], 'val_F1': [], 'val_F1_75': []
                      }
                      
# CREATE LABEL ENCODER/DECODER
label_utils = LabelUtils(network_size, num_classes, match_iou, ignore_iou, box_variances, anchor_boxes,
                        confidence_threshold, nms_iou_threshold, max_predictions)

# DEFINE MY OWN SCHEDULER
class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate, warmup_steps, decay_steps, alpha):
      self.initial_learning_rate = initial_learning_rate
      self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
      self.decay_steps = tf.cast(decay_steps, dtype=tf.float32)
      self.alpha = alpha

  def __call__(self, step):
      step = tf.cast(step, dtype=tf.float32)
      warmup_lr = self.initial_learning_rate * (step / self.warmup_steps) + 0.05 * self.initial_learning_rate
      step = tf.math.minimum(step, self.decay_steps)
      cosine_decay = 0.5 * (1. + tf.math.cos(math.pi * step / self.decay_steps))
      cosine_lr = ((1. - self.alpha) * cosine_decay + self.alpha) * self.initial_learning_rate
      return tf.math.minimum(warmup_lr, cosine_lr)


# LEARNING RATE SCHEDULER
if lr_scheduler == 'piecewise':
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=learning_rate_boundaries, values=learning_rates)
elif lr_scheduler == 'cosine':
    learning_rate_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=learning_rate,
                                                        decay_steps=train_epoch_steps * t_max, alpha=cosine_alpha)
elif lr_scheduler == 'mine':
    learning_rate_fn = MyLRSchedule(initial_learning_rate=learning_rate, warmup_steps=train_epoch_steps * warmup_epochs,
                                    decay_steps=train_epoch_steps * t_max, alpha=cosine_alpha)
elif lr_scheduler == 'rop':
    learning_rate_fn = learning_rate
    CHANGE_LR_BASED_ON_PEALATEU = True

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():

    # DEFINING LOSS AND ACCURACY
    loss_fn = Loss(num_classes, anchor_boxes, box_variances)
    accuracy_fn = AccuracyF1(num_classes, anchor_boxes, box_variances).compute_accuracy
        
    # CREATE MODEL
    model = create_mobilenet_ssdlite(input_shape=(network_size, network_size, 3), weight_decay=weight_decay,
                                    num_class=num_classes, num_priors=anchor_per_grid, trainable=trainable_backbone)
        
    # RESUME THE MODEL (FINE-TUNE Stage 2)
    if resume:
        model.load_weights(resume_checkpoint)
        model.trainable = True

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=momentum)
    # optimizer = tf.optimizers.Adam(learning_rate=learning_rate_fn)

    # DEFINE MAIN LOSS + L2 REG LOSS
    def main_loss(labels, predictions):
        # Use tf.reduce_mean() or tf.reduce_sum() or tf.nn.compute_average_loss()
        pll, pcl, ncl, pred_loss = tf.unstack(loss_fn(labels, predictions))
        # pll, pcl, ncl, pred_loss = loss_fn(labels, predictions)
        scaled_loss = tf.nn.compute_average_loss(pred_loss, global_batch_size=batch_size)
        scaled_pll = tf.nn.compute_average_loss(pll, global_batch_size=batch_size)
        scaled_pcl = tf.nn.compute_average_loss(pcl, global_batch_size=batch_size)
        scaled_ncl = tf.nn.compute_average_loss(ncl, global_batch_size=batch_size)

        reg_loss = [tf.nn.l2_loss(var) * weight_decay for var in model.trainable_variables if "/kernel" in var.name]
        # reg_loss = reg_loss + [tf.nn.l2_loss(var) * 0.01 * weight_decay for var in model.trainable_variables if "depthwise_kernel" in var.name]
        # reg_loss = reg_loss + [tf.nn.l2_loss(1. - var) * 0.1 * weight_decay for var in model.trainable_variables if "gamma" in var.name]
        # reg_loss = reg_loss + [tf.nn.l2_loss(var) * 0.1 * weight_decay for var in model.trainable_variables if "beta" in var.name]
        
        reg_loss = tf.nn.scale_regularization_loss(tf.math.add_n(reg_loss))

        return scaled_pll, scaled_pcl, scaled_ncl, reg_loss, scaled_loss, scaled_loss  # + reg_loss
        

# PREPARE CHECKPOINT FOLDER
if not os.path.exists(checkpoints_folder):
    os.mkdir(checkpoints_folder)

if len(os.listdir(checkpoints_folder)) > 0:
    os.system('rm -r ' + checkpoints_folder + '/*')

# CREATE TF DATA DATASET WITH MAP AND MULTI-THREAD READING
train_data_information = pickle.load(open(train_dict_address, "rb"))
random.shuffle(train_data_information)
train_data = []
train_labels = []

for data, label, augment_type in train_data_information:
    train_data.append(data)
    train_labels.append(np.array(label).astype('float32'))

train_labels = tf.ragged.constant(train_labels)

print("Total Number of Final Train Images = " + str(len(train_data)))

validation_data_information = pickle.load(open(validation_dict_address, "rb"))
validation_data = []
validation_labels = []

for data, label, augment_type in validation_data_information:
    validation_data.append(os.path.join(data))
    validation_labels.append(label.astype('float32'))

validation_labels = tf.ragged.constant(validation_labels)

print("Total Number of Final Validation Images = " + str(len(validation_data)))

# CREATE TRAIN DATASET
autotune = tf.data.experimental.AUTOTUNE
train_dataset_epochs = int(epochs * (train_epoch_steps / (len(train_data) / batch_size))) + 1
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_dataset = train_dataset.cache().repeat(train_dataset_epochs).shuffle(min(len(train_data), 50000), reshuffle_each_iteration=True)
train_dataset = train_dataset.map(label_utils.preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
train_dataset = train_dataset.prefetch(autotune)
train_dist_dataset = iter(mirrored_strategy.experimental_distribute_dataset(train_dataset))

# CREATE VALIDATION DATASET
validation_step_per_epoch = int(len(validation_data) / batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels))
val_dataset = val_dataset.cache().repeat(epochs + 2)
val_dataset = val_dataset.map(label_utils.preprocess_data_validation, num_parallel_calls=autotune)
val_dataset = val_dataset.batch(batch_size=batch_size, drop_remainder=False)
val_dataset = val_dataset.prefetch(autotune)
validation_dist_dataset = iter(mirrored_strategy.experimental_distribute_dataset(val_dataset))

start_time = timeit.default_timer()

# DEFINE ALL THE METRICS
with mirrored_strategy.scope():
    train_pos_class_loss = tf.keras.metrics.Mean(name='train_pos_class_loss')
    train_neg_class_loss = tf.keras.metrics.Mean(name='train_neg_class_loss')
    train_localization_loss = tf.keras.metrics.Mean(name='train_localization_loss')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_l2_loss = tf.keras.metrics.Mean(name='train_l2_loss')
    train_recall = tf.keras.metrics.Mean(name='train_recall')
    train_precision = tf.keras.metrics.Mean(name='train_precision')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    train_accuracy_75 = tf.keras.metrics.Mean(name='train_accuracy_75')

    test_pos_class_loss = tf.keras.metrics.Mean(name='test_pos_class_loss')
    test_neg_class_loss = tf.keras.metrics.Mean(name='test_neg_class_loss')
    test_localization_loss = tf.keras.metrics.Mean(name='test_localization_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_recall = tf.keras.metrics.Mean(name='test_recall')
    test_precision = tf.keras.metrics.Mean(name='test_precision')
    test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')
    test_accuracy_75 = tf.keras.metrics.Mean(name='test_accuracy_75')


# DEFINING TRAIN STEP FUNCTION
# @tf.function()
def train_step(train_step_input):
    train_data_mini_batch, train_labels_mini_batch = train_step_input

    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(train_data_mini_batch, training=True)
        pll, pcl, ncl, l2l, loss, total_loss = main_loss(train_labels_mini_batch, predictions)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # new_vars = []
    # for var in model.trainable_variables:
    #     new_vars.append((1. - learning_rate * weight_decay) * var)
        
    # model.set_weights(new_vars)

    train_localization_loss(pll)
    train_pos_class_loss(pcl)
    train_neg_class_loss(ncl)
    train_loss(loss)
    train_l2_loss(l2l)

    recall, precision, accuracy, accuracy_75 = accuracy_fn(train_labels_mini_batch, predictions)
    train_recall(recall)
    train_precision(precision)
    train_accuracy(accuracy)
    train_accuracy_75(accuracy_75)

    return loss


# DEFINE TEST STEP FUNCTION
# @tf.function()
def test_step(test_step_input):
    test_data_mini_batch, test_labels_mini_batch = test_step_input

    # training=False is only needed if there are layers with different behavior during training versus inference (e.g. Dropout).
    predictions = model(test_data_mini_batch, training=False)
    val_pll, val_pcl, val_ncl, val_l2l, validation_loss, validation_total_loss = main_loss(test_labels_mini_batch, predictions)

    test_localization_loss(val_pll)
    test_pos_class_loss(val_pcl)
    test_neg_class_loss(val_ncl)
    test_loss(validation_loss)

    val_recall, val_precision, validation_accuracy, validation_accuracy_75 = accuracy_fn(test_labels_mini_batch, predictions)
    test_recall(val_recall)
    test_precision(val_precision)
    test_accuracy(validation_accuracy)
    test_accuracy_75(validation_accuracy_75)


# DEFINE DISTRIBUTED TRAIN STEP
@tf.function()
def distributed_train_step(batch_inputs):
    # print(batch_inputs)
    per_replica_losses = mirrored_strategy.run(train_step, args=(batch_inputs,))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


# DEFINE DISTRIBUTED TEST STEP
@tf.function()
def distributed_test_step(test_batch_inputs):
    return mirrored_strategy.run(test_step, args=(test_batch_inputs,))
    
    
# DECOUPLED WEIGHT DECAY
@tf.function()
def apply_weight_decay():
    effective_lr = learning_rate_fn(optimizer.iterations)
    for var in model.trainable_variables:
        if "/kernel" in var.name:  #  or "bias" in var.name
            var.assign_sub(effective_lr * weight_decay * var.read_value())                
        # elif "/depthwise" in var.name:  #  or "bias" in var.name
        #     var.assign_sub(effective_lr * 0.01 * weight_decay * var.read_value())


# WARMUP FOR REDUCE ON PELATEU
last_lr_reduced_epoch = 0
last_saved_epoch = 0
if CHANGE_LR_BASED_ON_PEALATEU:
    optimizer.learning_rate.assign(learning_rate * 0.1)
    print("Warmup Learning Rate = " + str(optimizer.learning_rate))

print("START TRAINING")
for epoch in range(epochs):
    
    if CHANGE_LR_BASED_ON_PEALATEU:
        if epoch == warmup_epochs:
            optimizer.learning_rate.assign(learning_rate)
            print("Back to Real Learning Rate = " + str(optimizer.learning_rate))        

    # TO DO (2 Stage Training Shit in 1 run)
    # if epoch == warmup_epochs:
    #     print("Now Fine-Tune the entire model")
        # model = create_mobilenet_yolo(input_shape=(network_size, network_size, 3),
        #                               num_class=num_classes, num_priors=anchor_per_grid, verbose=False)
                                      
        # weights_files = os.listdir(checkpoints_folder)
        # list.sort(weights_files)
        # model.load_weights(os.path.join(checkpoints_folder, weights_files[-1]))
    #     print(len(model.trainable_variables))  
    #     model.trainable = True
    #     optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=momentum)
    #     model._init_batch_counters()
    #     model._reset_compile_cache()
    #     with tf.GradientTape() as tape:
    #         tape.reset()
    #     print(len(model.trainable_variables))

    epoch_start_time = timeit.default_timer()

    # RESET THE METRICS
    train_pos_class_loss.reset_states()
    train_neg_class_loss.reset_states()
    train_localization_loss.reset_states()
    train_loss.reset_states()
    train_l2_loss.reset_states()
    train_recall.reset_states()
    train_precision.reset_states()
    train_accuracy.reset_states()
    train_accuracy_75.reset_states()

    test_pos_class_loss.reset_states()
    test_neg_class_loss.reset_states()
    test_localization_loss.reset_states()
    test_loss.reset_states()
    test_recall.reset_states()
    test_precision.reset_states()
    test_accuracy.reset_states()
    test_accuracy_75.reset_states()

    # TRAIN ON ALL OF THE DATA PACK
    tqdm_progress_bar = tqdm(range(train_epoch_steps), desc="Epoch:" + str(epoch + 1) + '/' + str(epochs),
                             bar_format='{desc}, Step:{n_fmt}/{total_fmt} |{bar:10}|{postfix}')
    for index in tqdm_progress_bar:
        # TRAIN ON ONE BATCH OF DATA
        batch_data = next(train_dist_dataset)
        distributed_train_step(batch_data)
        
        # DECOUPLED WEIGHT DECAY
        apply_weight_decay()
        
        """
        effective_lr = learning_rate_fn(optimizer.iterations)
        for var in model.trainable_variables:
            if "/kernel" in var.name:  #  or "bias" in var.name
                var.assign_sub(effective_lr * weight_decay * var.read_value())                
            elif "depthwise" in var.name or "bias" in var.name:  #  or "bias" in var.name
                var.assign_sub(effective_lr * 0.1 * weight_decay * var.read_value())
        """
        
        tqdm_progress_bar.set_postfix({"PCl": '{:1.3f}'.format(train_pos_class_loss.result()),
                                       "NCl": '{:1.3f}'.format(train_neg_class_loss.result()),
                                       "LLoss": '{:1.3f}'.format(train_localization_loss.result()),
                                       "Loss": '{:1.3f}'.format(train_loss.result()),
                                       "L2L": '{:1.3f}'.format(train_l2_loss.result()),
                                       "Rec": '{:1.3f}'.format(train_recall.result()),
                                       "Prec": '{:1.3f}'.format(train_precision.result()),
                                       "F1": '{:1.3f}'.format(train_accuracy.result()),
                                       "75": '{:1.3f}'.format(train_accuracy_75.result())
                                       })

    if epoch >= start_validating:

        validation_history['pos_loss'].append(train_pos_class_loss.result().numpy())
        validation_history['neg_loss'].append(train_neg_class_loss.result().numpy())
        validation_history['loc_loss'].append(train_localization_loss.result().numpy())
        validation_history['loss'].append(train_loss.result().numpy())
        validation_history['recall'].append(train_recall.result().numpy())
        validation_history['precision'].append(train_precision.result().numpy())
        validation_history['F1'].append(train_accuracy.result().numpy())
        validation_history['F1_75'].append(train_accuracy_75.result().numpy())

        # CALCULATE METRICS FOR ALL OF VALIDATION DATA
        for index in range(validation_step_per_epoch):
            # Test ON ONE BATCH OF VALIDATION DATA
            valid_batch_data = next(validation_dist_dataset)
            distributed_test_step(valid_batch_data)

        # PRINT VALIDATION STATS
        print("Epoch:" + str(epoch + 1) + '/' + str(epochs) +
              ": VPCl=" + '{:1.3f}'.format(test_pos_class_loss.result()) +
              ", VNCl=" + '{:1.3f}'.format(test_neg_class_loss.result()) +
              ", VCl=" + '{:1.3f}'.format(test_pos_class_loss.result() + test_neg_class_loss.result()) +
              ", VLLoss=" + '{:1.3f}'.format(test_localization_loss.result()) +
              ", ValLoss=" + '{:1.3f}'.format(test_loss.result()) +
              ", VRec=" + '{:1.3f}'.format(test_recall.result()) +
              ", VPrec=" + '{:1.3f}'.format(test_precision.result()) +
              ", VF1=" + '{:1.3f}'.format(test_accuracy.result()) +
              ", V75=" + '{:1.3f}'.format(test_accuracy_75.result())
              )

        # APPEND VALIDATION METRICS TO VALIDATION HISTORY
        validation_history['val_pos_loss'].append(test_pos_class_loss.result().numpy())
        validation_history['val_neg_loss'].append(test_neg_class_loss.result().numpy())
        validation_history['val_loc_loss'].append(test_localization_loss.result().numpy())
        validation_history['val_class_loss'].append(test_pos_class_loss.result().numpy() + test_neg_class_loss.result().numpy())
        validation_history['val_loss'].append(test_loss.result().numpy())
        validation_history['val_recall'].append(test_recall.result().numpy())
        validation_history['val_precision'].append(test_precision.result().numpy())
        validation_history['val_F1'].append(test_accuracy.result().numpy())
        validation_history['val_F1_75'].append(test_accuracy_75.result().numpy())

        # SAVE BEST CHECKPOINT WEIGHTS AND MODELS BASED ON LOSS AND ACCURACY
        if validation_history['val_loss'][-1] <= np.min(validation_history['val_loss']):
            model.save_weights(checkpoints_folder + '/model_weights_best_loss_' + str(epoch).zfill(4) + '.h5')
            tf.keras.models.save_model(model, checkpoints_folder + '/model_best_loss_' + str(epoch).zfill(4) + '.h5')
            last_saved_epoch = epoch

        if validation_history['val_F1'][-1] >= np.max(validation_history['val_F1']):
            model.save_weights(checkpoints_folder + '/model_weights_best_accuracy_' + str(epoch).zfill(4) + '.h5')
            tf.keras.models.save_model(model, checkpoints_folder + '/model_best_accuracy_' + str(epoch).zfill(4) + '.h5')
            last_saved_epoch = epoch
            
        # LEARNING RATE SCHEDULER
        # REDUCE LEARNING RATE ON PELATU
        if CHANGE_LR_BASED_ON_PEALATEU and epoch - last_lr_reduced_epoch >= reduce_on_plt_patients:
            if epoch - last_saved_epoch > reduce_on_plt_patients:
                # if learning_rate * learning_rate_factor >= min_learning_rate:
                learning_rate = learning_rate * learning_rate_factor
                optimizer.learning_rate.assign(learning_rate)
                last_lr_reduced_epoch = epoch
                print("LR Reduce on Pelatu Happend With Learning Rate = " + str(optimizer.learning_rate))

        # SAVE SOME CHECKPOINTS FOR NO REASON
        if epoch % 2 == 0:
            model.save_weights(checkpoints_folder + '/model_weights_checkpoint_' + str(epoch).zfill(4) + '.h5')
            # tf.keras.models.save_model(model, checkpoints_folder + '/model_checkpoint_' + str(epoch).zfill(4) + '.h5')

        # SAVE THE VALIDATION HISTORY
        if epoch % 3 == 0:
            with open(checkpoints_folder + '/model_history_' + str(epoch).zfill(4) + '.pkl', 'wb') as file_pi:
                pickle.dump(validation_history, file_pi)

        # EARLY STOPPING (DELETE OR NEED)
        """
        if len(validation_history['val_loss']) > patience:
            if validation_history['val_loss'][-1] > np.min(validation_history['val_loss']) and len(
                    validation_history['val_loss']) - np.argmin(validation_history['val_loss']) >= patience:
                done_training_loss = True

        if len(validation_history['val_F1']) > patience:
            if validation_history['val_F1'][-1] < np.max(validation_history['val_F1']) and len(
                    validation_history['val_F1']) - np.argmax(validation_history['val_F1']) >= patience:
                done_training_accuracy = True

        # CHECK FOR EARLY STOPPING
        if done_training_loss and done_training_accuracy:
            break
        """

    print("Time of Epoch = " + str(round((timeit.default_timer() - epoch_start_time))) + " Seconds")

print("Time of All Run = " + str(round((timeit.default_timer() - start_time) / 60)) + " Minutes")

# PRINT ALL LOSS AND ACCURACY HISTORY
print("Validation Loss History = " + str(validation_history['val_loss']))
print("Validation Accuracy History = " + str(validation_history['val_F1']))

print("Minimum of Validation Loss History = " + str(np.min(validation_history['val_loss'])))
print("Maximum of Validation Accuracy History = " + str(np.max(validation_history['val_F1'])))






