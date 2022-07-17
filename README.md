# MobileDenseNet

## About code

This repository belong to our article at (article link)

Object Detection on COCO Dataset for Embedded Devices using Tensorflow 2.x from scratch, achieved 24.8 AP on COCO Dataset
![image](https://user-images.githubusercontent.com/11835873/179398211-906f4a8b-b2a8-4740-9307-473778a4d79f.png)


You only need Python3.7, Tensorflow2.3 and OpenCV for our code to work

## How to Train 

First download coco 2017 dataset from https://cocodataset.org/#home

Then create data list using data_reader_coco.py, before running change addresses to your locally coco dataset

``` python data_reader_coco.py ```

Then you can train using created pickle files

Our code also automaticly supports multi-gpu training

If you want better results you can use our pre_data_augmentation.py code to create more data and use cutmix and mix4 augmentation

## How to Run

You can use this tiny code for using our pre-trained model

```

# create anchor boxes
anchor_boxes = tf.convert_to_tensor(create_all_anchor_boxes(network_size, anchor_strides, anchor_areas,
                                                            anchor_scales, anchor_aspect_ratios), dtype=tf.float32)
                                                            
# create label encoder/decoder
label_utils = LabelUtils(network_size, num_classes, match_iou, ignore_iou, box_variances, anchor_boxes,
                        confidence_threshold, nms_iou_threshold, max_predictions)

# load the model
model = tf.keras.models.load_model(latest_checkpoint, compile=False)

# load the image
img = tf.io.read_file(file_address)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, (network_size, network_size))
ratio = [float(img.shape[0]) / float(real_image.shape[0]), float(img.shape[1]) / float(real_image.shape[1])]
img = preprocess_single_image(img)

# inference from model
model_output = model(tf.convert_to_tensor(validation_data), training=False)

# post-process (NMS and others)
extracted_preds, counter_preds = label_utils.extract_detections_from_predictions_argmax(model_output)

```


