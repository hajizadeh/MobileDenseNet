import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Reshape, ReLU, BatchNormalization,\
    DepthwiseConv2D, SeparableConv2D, SpatialDropout2D, Add, UpSampling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model


def create_mobilenet_ssdlite(input_shape=(320, 320, 3), num_priors=[6, 10, 10, 10], num_class=80, head_conv_filters=256, weight_decay=0.0001, trainable=True, verbose=True):

    my_input = Input(input_shape)
    base_model = MobileNet(input_shape=input_shape, weights="imagenet", include_top=False)
    # base_model.summary()
    
    # for layer in base_model.layers:
    #     if layer.name == "conv_dw_4":  # conv_dw_4
    #         break
    #     print(layer.name)
    #     layer.trainable=False
        
    if not trainable:
        base_model.trainable = trainable
        
    my_base_model = Model(base_model.inputs, [base_model.get_layer('conv_pw_5_relu').output,
                                              base_model.get_layer('conv_pw_11_relu').output,   # base_model.get_layer('conv_pw_5_relu').output,
                                              base_model.get_layer('conv_pw_13_relu').output])
    # Getting c3 c4 and c5 from backbone
    [c3, c4, c5] = my_base_model(my_input)  # , training=False

    feature_pyramids = []

    # P3
    p3 = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False, name='p3')(c3)
    p3 = BatchNormalization()(p3)
    p3 = ReLU(max_value=6.0)(p3)

    # P4
    p4 = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False, name='p4')(c4)
    p4 = BatchNormalization()(p4)
    p4 = ReLU(max_value=6.0)(p4)

    # P5
    p5 = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False, name='p5')(c5)
    p5 = BatchNormalization()(p5)
    p5 = ReLU(max_value=6.0)(p5)

    
    # NECK
    p4 = Add()([UpSampling2D(size=(2, 2))(p5), p4])
    
    p4 = DepthwiseConv2D(3, 1, padding='same', use_bias=False, name='p4_dw')(p4)
    p4 = BatchNormalization()(p4)
    p4 = ReLU(max_value=6.0)(p4)
    p4 = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False, name='p4_pw')(p4)
    p4 = BatchNormalization()(p4)
    p4 = ReLU(max_value=6.0)(p4)
    p4 = DepthwiseConv2D(3, 1, padding='same', use_bias=False, name='p42_dw')(p4)
    p4 = BatchNormalization()(p4)
    p4 = ReLU(max_value=6.0)(p4)
    p4 = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False, name='p42_pw')(p4)
    p4 = BatchNormalization()(p4)
    p4 = ReLU(max_value=6.0)(p4)
    
    p3 = Add()([UpSampling2D(size=(2, 2))(p4), p3])
    
    p3 = DepthwiseConv2D(3, 1, padding='same', use_bias=False, name='p3_dw')(p3)
    p3 = BatchNormalization()(p3)
    p3 = ReLU(max_value=6.0)(p3)
    p3 = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False, name='p3_pw')(p3)
    p3 = BatchNormalization()(p3)
    p3 = ReLU(max_value=6.0)(p3)
    p3 = DepthwiseConv2D(3, 1, padding='same', use_bias=False, name='p32_dw')(p3)
    p3 = BatchNormalization()(p3)
    p3 = ReLU(max_value=6.0)(p3)
    p3 = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False, name='p32_pw')(p3)
    p3 = BatchNormalization()(p3)
    p3 = ReLU(max_value=6.0)(p3)
    
    
    feature_pyramids.append(p3)

    feature_pyramids.append(p4)

    feature_pyramids.append(p5)

    # Creating P6 and P7 from backbone ouptput (P5)
    # P6
    p6 = DepthwiseConv2D(3, 2, padding='same', use_bias=False, name='p6_dw')(p5)
    p6 = BatchNormalization()(p6)
    p6 = ReLU(max_value=6.0)(p6)
    p6 = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False, name='p6_pw')(p6)
    p6 = BatchNormalization()(p6)
    p6 = ReLU(max_value=6.0)(p6)
    
    p6 = DepthwiseConv2D(3, 1, padding='same', use_bias=False, name='p62_dw')(p6)
    p6 = BatchNormalization()(p6)
    p6 = ReLU(max_value=6.0)(p6)
    p6 = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False, name='p62_pw')(p6)
    p6 = BatchNormalization()(p6)
    p6 = ReLU(max_value=6.0)(p6)
    
    feature_pyramids.append(p6)

    """
    # P7
    p7 = DepthwiseConv2D(3, 2, padding='same', use_bias=False, name='p7_dw')(p6)
    p7 = BatchNormalization()(p7)
    p7 = ReLU(max_value=6.0)(p7)
    p7 = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False, name='p7_pw')(p7)
    p7 = BatchNormalization()(p7)
    p7 = ReLU(max_value=6.0)(p7)
    
    p7 = DepthwiseConv2D(3, 1, padding='same', use_bias=False, name='p72_dw')(p7)
    p7 = BatchNormalization()(p7)
    p7 = ReLU(max_value=6.0)(p7)
    p7 = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False, name='p72_pw')(p7)
    p7 = BatchNormalization()(p7)
    p7 = ReLU(max_value=6.0)(p7)
    
    feature_pyramids.append(p7)
    """
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    prior_prob = -np.log((1. - 0.01) / 0.01)
    # bias_init = tf.constant_initializer(np.tile(np.concatenate([np.tile(prior_prob, num_class), np.array([-prior_prob])], axis=0), num_priors[idx]))

    # Head
    predictions = []
    for idx, fp in enumerate(feature_pyramids):
        # CLASS HEAD
        fp_head_class = DepthwiseConv2D(3, 1, padding='same', use_bias=False)(fp)
        fp_head_class = BatchNormalization()(fp_head_class)
        fp_head_class = ReLU(max_value=6.0)(fp_head_class)
        fp_head_class = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False)(fp_head_class)
        fp_head_class = BatchNormalization()(fp_head_class)
        fp_head_class = ReLU(max_value=6.0)(fp_head_class)
        fp_head_class = DepthwiseConv2D(3, 1, padding='same', use_bias=False)(fp_head_class)
        fp_head_class = BatchNormalization()(fp_head_class)
        fp_head_class = ReLU(max_value=6.0)(fp_head_class)
        fp_head_class = Conv2D(head_conv_filters, 1, 1, padding='same', use_bias=False)(fp_head_class)
        fp_head_class = BatchNormalization()(fp_head_class)
        fp_head_class = ReLU(max_value=6.0)(fp_head_class)
        
        # BBOX HEAD
        fp_head_box = DepthwiseConv2D(3, 1, padding='same', use_bias=False)(fp)
        fp_head_box = BatchNormalization()(fp_head_box)
        fp_head_box = ReLU(max_value=6.0)(fp_head_box)
        fp_head_box = Conv2D(int(head_conv_filters / 2), 1, 1, padding='same', use_bias=False)(fp_head_box)
        fp_head_box = BatchNormalization()(fp_head_box)
        fp_head_box = ReLU(max_value=6.0)(fp_head_box)
        fp_head_box = DepthwiseConv2D(3, 1, padding='same', use_bias=False)(fp_head_box)
        fp_head_box = BatchNormalization()(fp_head_box)
        fp_head_box = ReLU(max_value=6.0)(fp_head_box)
        fp_head_box = Conv2D(int(head_conv_filters / 2), 1, 1, padding='same', use_bias=False)(fp_head_box)
        fp_head_box = BatchNormalization()(fp_head_box)
        fp_head_box = ReLU(max_value=6.0)(fp_head_box)
        
        bias_init = tf.constant_initializer(np.tile(np.concatenate([np.tile(prior_prob, num_class), np.array([-prior_prob])], axis=0), num_priors[idx]))
        # bias_init = tf.constant_initializer(prior_prob)
            
        # Seperate Head
        fp_box_preds = Conv2D(num_priors[idx] * 4, 1, 1, padding="same")(fp_head_box)
        fp_box_preds_reshape = Reshape((-1, 4))(fp_box_preds)
        
        fp_class_preds = Conv2D(num_priors[idx] * (num_class + 1), 1, 1, bias_initializer=bias_init, padding="same")(fp_head_class)
        fp_class_preds_reshape = Reshape((-1, num_class + 1))(fp_class_preds)
        
        fp_preds_concat = Concatenate(axis=-1)([fp_box_preds_reshape, fp_class_preds_reshape])
        print(fp_preds_concat.shape)
        predictions.append(fp_preds_concat)

    all_predictions = Concatenate(axis=1)(predictions)

    mobilenet_ssdlite = Model(my_input, all_predictions)        
    
    """
    mobilenet_ssdlite.save_weights("tmp.h5")    
    l2_reg = tf.keras.regularizers.l2(weight_decay / 2)    
    decay_attributes = ['kernel_regularizer', 'bias_regularizer']  # 'kernel_regularizer', 'bias_regularizer', 'beta_regularizer', 'gamma_regularizer'
    for layer in mobilenet_ssdlite.layers:
        if isinstance(layer, tf.keras.models.Model):
            for sub_layer in layer.layers:
                for attr in decay_attributes:
                    if hasattr(sub_layer, attr) and sub_layer.trainable:
                        setattr(sub_layer, attr, l2_reg)        
        for attr in decay_attributes:
            if hasattr(layer, attr) and layer.trainable:
                setattr(layer, attr, l2_reg)
                
    mobilenet_ssdlite = tf.keras.models.model_from_json(mobilenet_ssdlite.to_json())
    mobilenet_ssdlite.load_weights("tmp.h5", by_name=True)  
    """     

    if verbose:
        mobilenet_ssdlite.summary()
        
    """
    print(tf.math.add_n(mobilenet_ssdlite.losses))
    reg_loss = [tf.nn.l2_loss(var) * 0.00004 for var in mobilenet_ssdlite.trainable_variables if ("gamma" in var.name)]
    print(tf.math.add_n(reg_loss))
    mean = [tf.reduce_mean(var) for var in mobilenet_ssdlite.trainable_variables if "gamma" in var.name]
    print(tf.reduce_mean(mean))
    """

    return mobilenet_ssdlite


if __name__ == '__main__':
    create_mobilenet_ssdlite()
