# from utils.imagenet_utils import _obtain_input_shape
# from tensorflow.keras import backend
# from tensorflow.keras import models 
# from tensorflow.keras import utils as keras_utils 
# from tensorflow.keras import layers


# def conv3x3(in_planes, out_planes, stride =1 , padding=1, dilation=1, name=None):
#     x = layers.Conv2D(out_planes, 3, strides=stride, padding=padding, dilation_rate=dilation,, name=name)
#     return x

# def BasicBlock(input, in_planes, planes, stride =1, downsample=None, dilation=(1,1), residual=True, name="basickblock"):
#     shortcut = input
#     out = conv3x3(in_planes, planes, stride, padding=dilation[0],dilation=dilation[0],name=name+"/conv1")(input)
#     out = layers.BatchNormalization(name=name+"/bn1")(out)
#     out = layers.ReLU(name=name+"/relu1")(out)
#     out = conv3x3(planes, planes, stride, padding=dilation[1],dilation=dilation[1], name=name+"/conv2")(out)
#     out = layers.BatchNormalization(name=name+"/bn2")(out)

#     if(downsample is not None):
#         shortcut = downsample(name=name+"/downsample1")(shortcut)
#     if(residual):
#         out  = layers.Add(name=name+"/add1")([out, shortcut])
#     out = layers.ReLU(name=name+"/relu2")(out)
#     return out

# def Bottleneck(input, in_planes, planes, stride=1, downsample=None, dilation=(1,1), residual=True,name="bottleneck"):
#     out = layers.Conv2D(out_planes, 1, strides=stride, padding=padding, dilation_rate=dilation,, name=name+"/conv1")


#ref:
# current author M.Ikhtiyor
# [1] https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

# import the necessary packages
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers 
import numpy as np 
import datetime as dt 
import os 
import warnings 
from utils.imagenet_utils import _obtain_input_shape
from tensorflow.keras import backend
from tensorflow.keras import models 
from tensorflow.keras import utils as keras_utils 

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

def identity_block(input_tensor, kernel_size, filters, stage, block, dilation = 1):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    ac_name_base = 'activation'+ str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a', )(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu',name=ac_name_base + '2a')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b', dilation_rate=dilation, )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu',name=ac_name_base + '2b')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c', )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu',name=ac_name_base + '2c')(x)
    return x



def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2), dilation = 1):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    ac_name_base = 'activation'+ str(stage) + block + '_branch'
    # layers.Conv2D(out_planes, 3, strides=stride, padding=padding, dilation_rate=dilation,, name=name)
    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu',name=ac_name_base + '2a')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b',  dilation_rate=dilation, )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu',name=ac_name_base + '2b')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c', )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1', )(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu',name=ac_name_base + '2c')(x)
    return x

def DRN50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1)) # dilation 1
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b') # dilation 1
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c') # dilation 1

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a') # dilation 1
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')# dilation 1
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')# dilation 1
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')# dilation 1

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', dilation=2,strides=(1, 1)) # dilation 2
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', dilation=2) # dilation 2
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', dilation=2) # dilation 2
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', dilation=2)# dilation 2
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', dilation=2)# dilation 2
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', dilation=2)# dilation 2

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', dilation=4,strides=(1, 1)) # dilation 4
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dilation=4) # dilation 4
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dilation=4) # dilation 4

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet50')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model
