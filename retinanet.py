import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Reshape, ReLU, BatchNormalization, Add, UpSampling2D
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Model


def create_retinaface(input_shape=(640, 640, 3), num_priors=[3, 3, 3, 3, 3, 3], num_class=1, trainable=True, verbose=True):

    my_input = Input(input_shape)
    base_model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    # base_model = MobileNetV2(input_shape=input_shape, alpha=0.5, weights='imagenet', include_top=False)
    # base_model.summary()
    # for layer in base_model.layers:
    #     print(layer.name)
    #     if "conv2" in layer.name:
    #         break
    #     layer.trainable = False
    
    if not trainable:
        base_model.trainable = trainable

    my_base_model = Model(base_model.inputs, [base_model.get_layer('conv2_block3_out').output,
                                              base_model.get_layer('conv3_block4_out').output,
                                              base_model.get_layer('conv4_block6_out').output,
                                              base_model.get_layer('conv5_block3_out').output])
    # Getting c3 c4 and c5 from backbone
    [c2, c3, c4, c5] = my_base_model(my_input, training=False)  # , training=False

    feature_pyramids = []
    
    # P2
    p2 = Conv2D(256, 1, 1, padding='same', name='p2')(c2)
    p2 = BatchNormalization()(p2)
    p2 = ReLU()(p2)

    # P3
    p3 = Conv2D(256, 1, 1, padding='same', name='p3')(c3)
    p3 = BatchNormalization()(p3)
    p3 = ReLU()(p3)

    # P4
    p4 = Conv2D(256, 1, 1, padding='same', name='p4')(c4)
    p4 = BatchNormalization()(p4)
    p4 = ReLU()(p4)

    # P5
    p5 = Conv2D(256, 1, 1, padding='same', name='p5')(c5)
    p5 = BatchNormalization()(p5)
    p5 = ReLU()(p5)

    # NECK
    p4 = Add()([UpSampling2D(size=(2, 2))(p5), p4])    
    p4 = Conv2D(256, 3, 1, padding='same', name='p4_1')(p4)
    p4 = BatchNormalization()(p4)
    p4 = ReLU()(p4)
    p4 = Conv2D(256, 3, 1, padding='same', name='p4_2')(p4)
    p4 = BatchNormalization()(p4)
    p4 = ReLU()(p4)
    
    p3 = Add()([UpSampling2D(size=(2, 2))(p4), p3])    
    p3 = Conv2D(256, 3, 1, padding='same', name='p3_1')(p3)
    p3 = BatchNormalization()(p3)
    p3 = ReLU()(p3)
    p3 = Conv2D(256, 3, 1, padding='same', name='p3_2')(p3)
    p3 = BatchNormalization()(p3)
    p3 = ReLU()(p3) 

    p2 = Add()([UpSampling2D(size=(2, 2))(p3), p2])    
    p2 = Conv2D(256, 3, 1, padding='same', name='p2_1')(p2)
    p2 = BatchNormalization()(p2)
    p2 = ReLU()(p2)
    p2 = Conv2D(256, 3, 1, padding='same', name='p2_2')(p2)
    p2 = BatchNormalization()(p2)
    p2 = ReLU()(p2)
    
    feature_pyramids.append(p2)
    
    feature_pyramids.append(p3)

    feature_pyramids.append(p4)

    feature_pyramids.append(p5)

    # Creating P6 and P7 from backbone ouptput (P5)
    # P6
    p6 = Conv2D(256, 3, 2, padding='same', name='p6')(p5)
    p6 = BatchNormalization()(p6)
    p6 = ReLU()(p6)
    p6 = Conv2D(256, 3, 1, padding='same', name='p6_2')(p6)
    p6 = BatchNormalization()(p6)
    p6 = ReLU()(p6)
    
    feature_pyramids.append(p6)
    """
    # P7
    p7 = Conv2D(256, 3, 2, padding='same', name='p7')(p6)
    p7 = BatchNormalization()(p7)
    p7 = ReLU()(p7)
    p7 = Conv2D(256, 3, 1, padding='same', name='p7_2')(p7)
    p7 = BatchNormalization()(p7)
    p7 = ReLU()(p7)
    
    feature_pyramids.append(p7)
    """
    
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    prior_probability = tf.constant_initializer([-np.log((1 - 0.1) / 0.1), np.log((1 - 0.1) / 0.1),   # , np.log((1 - 0.1) / 0.1),
                                                -np.log((1 - 0.1) / 0.1), np.log((1 - 0.1) / 0.1),
                                                -np.log((1 - 0.1) / 0.1), np.log((1 - 0.1) / 0.1)])
    
    # , kernel_initializer=kernel_init
    # , bias_initializer=prior_probability
    
    # Define Share Layers for SSH and Head
    

    # Head
    predictions = []
    for idx, fp in enumerate(feature_pyramids):
        """    
        fp_head_1 = Conv2D(128, 3, 1, padding='same')(fp)
        fp_head_1 = BatchNormalization()(fp_head_1)
        
        fp_head = Conv2D(128, 3, 1, padding='same')(fp)
        fp_head = BatchNormalization()(fp_head)
        fp_head = ReLU()(fp_head)        
        fp_head_2 = Conv2D(64, 3, 1, padding='same')(fp_head)
        fp_head_2 = BatchNormalization()(fp_head_2)
        
        fp_head_3 = Conv2D(64, 3, 1, padding='same')(fp_head)
        fp_head_3 = BatchNormalization()(fp_head_3)
        fp_head_3 = ReLU()(fp_head_3)        
        fp_head_3 = Conv2D(64, 3, 1, padding='same')(fp_head_3)
        fp_head_3 = BatchNormalization()(fp_head_3)
        
        final_fp_head = Concatenate(axis=-1)([fp_head_1, fp_head_2, fp_head_3])
        final_fp_head = ReLU()(final_fp_head) 
        
        
        fp_head_class_1 = Conv2D(128, 3, 1, padding='same')(fp)
        fp_head_class_1 = BatchNormalization()(fp_head_class_1)
        
        fp_head_class = Conv2D(128, 3, 1, padding='same')(fp)
        fp_head_class = BatchNormalization()(fp_head_class)
        fp_head_class = ReLU()(fp_head_class)        
        fp_head_class_2 = Conv2D(64, 3, 1, padding='same')(fp_head_class)
        fp_head_class_2 = BatchNormalization()(fp_head_class_2)
        
        fp_head_class_3 = Conv2D(64, 3, 1, padding='same')(fp_head_class)
        fp_head_class_3 = BatchNormalization()(fp_head_class_3)
        fp_head_class_3 = ReLU()(fp_head_class_3)        
        fp_head_class_3 = Conv2D(64, 3, 1, padding='same')(fp_head_class_3)
        fp_head_class_3 = BatchNormalization()(fp_head_class_3)
        
        final_fp_head_class = Concatenate(axis=-1)([fp_head_class_1, fp_head_class_2, fp_head_class_3])
        final_fp_head_class = ReLU()(final_fp_head_class) 

        
        # Seperate Head
        fp_box_preds = Conv2D(num_priors[idx] * 4, 3, 1, padding="same")(final_fp_head)
        fp_box_preds_reshape = Reshape((-1, 4))(fp_box_preds)
        
        fp_class_preds = Conv2D(num_priors[idx] * (num_class + 1), 3, 1, padding="same", bias_initializer=prior_probability)(final_fp_head)
        fp_class_preds_reshape = Reshape((-1, (num_class + 1)))(fp_class_preds)
        """
        
        """
        fp_head = Conv2D(256, 3, 1, padding='same')(fp)
        fp_head = BatchNormalization()(fp_head)
        fp_head = ReLU()(fp_head)
        fp_head = Conv2D(256, 3, 1, padding='same')(fp_head)
        fp_head = BatchNormalization()(fp_head)
        fp_head = ReLU()(fp_head)
        """
        fp = Conv2D(256, 3, 1, padding='same')(fp)
        fp = BatchNormalization()(fp)
        fp = ReLU()(fp)
        
        ssh1 = Conv2D(128, 3, 1, padding='same')(fp)
        ssh1 = BatchNormalization()(ssh1)
        ssh1 = ReLU()(ssh1)
        
        ssh2 = Conv2D(64, 3, 1, padding='same')(ssh1)
        ssh2 = BatchNormalization()(ssh2)
        ssh2 = ReLU()(ssh2)
        
        ssh3 = Conv2D(64, 3, 1, padding='same')(ssh2)
        ssh3 = BatchNormalization()(ssh3)
        ssh3 = ReLU()(ssh3)
        
        fp_head = Concatenate(axis=-1)([ssh1, ssh2, ssh3])
        
        # # Predictions, Joint Head
        # fp_preds = Conv2D(num_priors[idx] * (4 + num_class + 1), 1, 1, padding="same")(fp_head)
        # fp_preds_reshape = Reshape((-1, (4 + num_class + 1)))(fp_preds)
        
        # Predictions, Seperate Head
        fp_box_preds = Conv2D(num_priors[idx] * 4, 1, 1, padding="same")(fp_head)
        fp_box_preds_reshape = Reshape((-1, 4))(fp_box_preds)
        
        fp_class_preds = Conv2D(num_priors[idx] * (num_class + 1), 1, 1, padding="same")(fp_head)
        fp_class_preds_reshape = Reshape((-1, (num_class + 1)))(fp_class_preds)
        
        fp_preds_concat = Concatenate(axis=-1)([fp_box_preds_reshape, fp_class_preds_reshape])
        
        print(fp_preds_concat.shape)
        predictions.append(fp_preds_concat)

    all_predictions = Concatenate(axis=1)(predictions)

    retinaface = Model(my_input, all_predictions)

    if verbose:
        retinaface.summary()

    return retinaface


if __name__ == '__main__':
    create_retinaface()
