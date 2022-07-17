import os
import pickle
import random
import cv2
import timeit
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from create_default_boxes import create_all_anchor_boxes
from config import network_size, num_classes, match_iou, ignore_iou, confidence_threshold, nms_iou_threshold, max_predictions
from config import anchor_strides, anchor_areas, anchor_scales, anchor_aspect_ratios, anchor_per_grid, box_variances
from utils import preprocess_single_image

from label_utils import LabelUtils


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# DEFINE ALL DATASET ARGUMENTS HERE
ground_truth_address = '/home/mohammad.hajizadeh/coco_object_detection/object_detection_metrics/my_groundtruths/'
detections_address = '/home/mohammad.hajizadeh/coco_object_detection/object_detection_metrics/my_detections/'
DATASET_LIST = "coco_val.pkl"
OBJECT_DICT_ADDRESS = 'object_names_coco.pkl'

# MODEL FOLDER AND WEIGHTS FILE
folder_address = "checkpoints_mobilenet_best/"
latest_checkpoint = folder_address + "model_best_accuracy_0228.h5"
number_of_test_images = 4592  # 4592
print(latest_checkpoint)

# CREATE ANCHOR BOXES
anchor_boxes = tf.convert_to_tensor(create_all_anchor_boxes(network_size, anchor_strides, anchor_areas,
                                                            anchor_scales, anchor_aspect_ratios), dtype=tf.float32)
                                                            
# CREATE LABEL ENCODER/DECODER
label_utils = LabelUtils(network_size, num_classes, match_iou, ignore_iou, box_variances, anchor_boxes,
                        confidence_threshold, nms_iou_threshold, max_predictions)

PRINT_OUTPUT_MODE = False
if PRINT_OUTPUT_MODE:
    label_utils.confidence_threshold = 0.45
    number_of_test_images = 50 * 16

if len(os.listdir(ground_truth_address)) > 0:
    os.system('rm -r ' + ground_truth_address)
    os.mkdir(ground_truth_address)

if len(os.listdir(detections_address)) > 0:
    os.system('rm -r ' + detections_address)
    os.mkdir(detections_address)

validation_data_information = pickle.load(open(DATASET_LIST, "rb"))
if PRINT_OUTPUT_MODE:
    random.seed(43)
    random.shuffle(validation_data_information)

print("Total Number of Validation Images = " + str(len(validation_data_information)))
print("Total Number Sellected Images for Test = " + str(number_of_test_images))

if PRINT_OUTPUT_MODE:
    if not os.path.exists(folder_address + 'output_images/'):
        os.system('mkdir ' + folder_address + 'output_images/')
    elif len(os.listdir(folder_address + 'output_images/')) > 0:
        os.system('rm ' + folder_address + 'output_images/*')
        
# VERY IMPORTANT FUNCTION
object_dict = pickle.load(open(OBJECT_DICT_ADDRESS, "rb"))
reverse_object_dict = {}
for key, value in object_dict.items():
    reverse_object_dict[value] = key

# LOAD THE MODEL
model = tf.keras.models.load_model(latest_checkpoint, compile=False)
reg_loss = [tf.nn.l2_loss(var) * 0.001 for var in model.trainable_variables if "/kernel" in var.name]
print(tf.math.add_n(reg_loss))
reg_loss = [tf.nn.l2_loss(var) * 0.1 * 0.001 for var in model.trainable_variables if ("depthwise_kernel" in var.name or "bias" in var.name)]
print(tf.math.add_n(reg_loss))

# READ IMAGES ONE BY ONE
validation_data = []
labels = []
image_names = []
real_image_sizes = []
real_images = []
ratios = []
real_predictions = []

decode_start_time = timeit.default_timer()
batch_size = 8
prediction_counter = []

tqdm_progress = tqdm(validation_data_information[:number_of_test_images], desc="Evaluating Images:", bar_format='{desc} Step:{n_fmt}/{total_fmt} |{bar:30}|{postfix}')
for data, label, augment_type in tqdm_progress:
    file_address = os.path.join(data)

    real_image = cv2.imread(file_address)

    img = tf.io.read_file(file_address)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (network_size, network_size))
    ratios.append([float(img.shape[0]) / float(real_image.shape[0]), float(img.shape[1]) / float(real_image.shape[1])])
    img = preprocess_single_image(img)

    validation_data.append(img)
    labels.append(label)
    image_names.append(data)
    real_image_sizes.append(real_image.shape[:2])
    if PRINT_OUTPUT_MODE:
        real_images.append(real_image)
    
    if len(validation_data) == batch_size:
        model_output = model(tf.convert_to_tensor(validation_data), training=False)
        extracted_preds, counter_preds = label_utils.extract_detections_from_predictions_argmax(model_output)
        real_predictions.extend(extracted_preds)
        validation_data = []
        prediction_counter.extend(counter_preds)
        tqdm_progress.set_postfix({"Predictions per image": np.mean(prediction_counter)})
        
if len(validation_data) > 0:
    model_output = model(tf.convert_to_tensor(validation_data), training=False)
    extracted_preds, counter_preds = label_utils.extract_detections_from_predictions_argmax(model_output)
    real_predictions.extend(extracted_preds)
    validation_data = []
    prediction_counter.extend(counter_preds)

print(len(real_predictions))
print("Detection Time = " + str(timeit.default_timer() - decode_start_time))
print(type(real_predictions))
write_start_time = timeit.default_timer()

for idx, img_size in enumerate(real_image_sizes):

    one_image_predictions = real_predictions[idx]
    # print(np.shape(one_image_predictions))
    ground_truth = ""
    predictions = ""
    if PRINT_OUTPUT_MODE:
        real_image = real_images[idx]
        
    for gt in labels[idx]:
        x_min = (gt[0] - (gt[2] / 2)) * float(img_size[1])
        y_min = (gt[1] - (gt[3] / 2)) * float(img_size[0])
        x_max = (gt[0] + (gt[2] / 2)) * float(img_size[1])
        y_max = (gt[1] + (gt[3] / 2)) * float(img_size[0])
        # label_name = get_object_name(gt[4])
        label_name = reverse_object_dict[gt[4]]
        # label_name = object_names_human_read[label_name]

        # if PRINT_OUTPUT_MODE:
        #     cv2.rectangle(real_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), thickness=2)
            
            # (text_width, text_height) = cv2.getTextSize(str(label_name), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1)[0]
            # cv2.rectangle(real_image, (int((x_min + x_max) * 0.5 - text_width / 2), int((y_min + y_max) * 0.5 - text_height / 2)),
            #                           (int((x_min + x_max) * 0.5 + text_width / 2 + 10), int((y_min + y_max) * 0.5 + text_height / 2 + 10)), (0, 0, 255), thickness=-2)
            # cv2.putText(real_image, str(label_name),
            #             (int((x_min + x_max) * 0.5 - text_width / 2 + 5), int((y_min + y_max) * 0.5 + text_height / 2 + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        ground_truth = ground_truth + str(label_name).replace(" ", "_")
        ground_truth = ground_truth + " " + str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max) + "\n"

    with open(ground_truth_address + image_names[idx][-35:-3].replace("/", "_") + 'txt', "w") as file:
        file.write(ground_truth)
    
    for pred_box in one_image_predictions:        
        x_min = np.clip((pred_box[2] - pred_box[4]/2) * float(img_size[1]), 0.0, float(img_size[1]))
        y_min = np.clip((pred_box[3] - pred_box[5]/2) * float(img_size[0]), 0.0, float(img_size[0]))
        x_max = np.clip((pred_box[2] + pred_box[4]/2) * float(img_size[1]), 0.0, float(img_size[1]))
        y_max = np.clip((pred_box[3] + pred_box[5]/2) * float(img_size[0]), 0.0, float(img_size[0]))
        
        score = pred_box[1]  # .numpy()
        label = int(pred_box[0])
        # label_name = get_object_name(label)
        label_name = reverse_object_dict[label]
        # label_name = object_names_human_read[label_name]

        # if (y_max - y_min) * (x_max - x_min) < MIN_AREA:
        #     continue

        if PRINT_OUTPUT_MODE:
            np.random.seed(label * 2 + 42)
            color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
            (text_width, text_height) = cv2.getTextSize(str(label_name) + ": " + str(np.round(score, 2)),
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        fontScale=0.6, thickness=1)[0]
            cv2.rectangle(real_image, (int(x_min), int(y_min - text_height - 10)), (int(x_min + text_width + 10), int(y_min)), color,
                          thickness=-2)

            cv2.rectangle(real_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness=2)

            cv2.putText(real_image, str(label_name) + ": " + str(np.round(score, 2)),
                        (int(x_min + 5), int(y_min - text_height + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        predictions = predictions + str(label_name).replace(" ", "_")
        predictions = predictions + " " + str(score) + " " + str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max) + "\n"

    with open(detections_address + image_names[idx][-35:-3].replace("/", "_") + 'txt', "w") as file:
        file.write(predictions)

    if PRINT_OUTPUT_MODE:
        cv2.imwrite(folder_address + 'output_images/' + image_names[idx][-35:-4].replace("/", "_") + '.jpg', real_image)
        
print("Wrint Files Time = " + str(timeit.default_timer() - write_start_time))

if not PRINT_OUTPUT_MODE:
    os.system("python object_detection_metrics/pascalvoc.py -gt " + ground_truth_address + " -det " + detections_address + " -np")

if not PRINT_OUTPUT_MODE:
    result_folder = "results_" + latest_checkpoint[-5:-3] + "/"
    if os.path.exists(folder_address + result_folder):
        os.system('rm -r ' + folder_address + result_folder + '*')
    else:
        os.mkdir(folder_address + result_folder)    
    os.system('cp -r object_detection_metrics/results/* ' + folder_address + result_folder)
    
print(latest_checkpoint)






















