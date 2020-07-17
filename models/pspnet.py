# from keras_applications import get_submodules_from_kwargs

from models import Conv2dBn
from utils import freeze_model
from utils import resize_bilinear
from models.backbones import DRN50, DRN101
from models.backbones import BackboneUtils
from tensorflow.keras import backend
from tensorflow.keras import models 
from tensorflow.keras import layers
from tensorflow.keras import utils as keras_utils 
import os 

# selecting the name of our backbone model
backbone_utils = None

def check_input_shape(input_shape, zoom_factor):
    if input_shape is None:
        raise ValueError("Input shape should be a tuple of 3 integers, not None!")

    h, w = input_shape[:2] if backend.image_data_format() == 'channels_last' else input_shape[1:]
    print('input shape is ', (h, w))
    # size-1 should be dividable by 8 for example 33
    assert (h-1) % 8 == 0 and (w-1) % 8 == 0   

# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv1x1BnReLU(filters, use_batchnorm, name=None):

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
        )(input_tensor)

    return wrapper
def cls(filters, use_batchnorm,n_classes,dropout=None):
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        
    def wrapper(input_tensor):
        
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='cls/zeropad1')(input_tensor)
        x = layers.Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                padding='valid',
                kernel_initializer='glorot_uniform',
                name='cls/conv1',
            )(x)
        if(use_batchnorm):
            x = layers.BatchNormalization(axis=bn_axis, name='cls/bn1')(x)
        x = layers.Activation('relu', name='cls/relu1')(x)
        # model regularization
        if dropout is not None:
            x = layers.SpatialDropout2D(dropout, name='cls/drp1')(x)
        
        # model head
        x = layers.Conv2D(
                filters=n_classes,
                kernel_size=(1, 1),
                padding='valid',
                kernel_initializer='glorot_uniform',
                name='final_conv',
            )(x)
        return x
    return wrapper


def SpatialContextBlock(
        level,
        conv_filters=512,
        pooling_type='avg',
        use_batchnorm=True,
):
    if pooling_type not in ('max', 'avg'):
        raise ValueError('Unsupported pooling type - `{}`.'.format(pooling_type) +
                         'Use `avg` or `max`.')

    Pooling2D = layers.MaxPool2D if pooling_type == 'max' else layers.AveragePooling2D

    pooling_name = 'psp_level{}_pooling'.format(level)
    conv_block_name = 'psp_level{}'.format(level)
    upsampling_name = 'psp_level{}_upsampling'.format(level)

    def wrapper(input_tensor):
        # extract input feature maps size (h, and w dimensions)
        input_shape = backend.int_shape(input_tensor)
        spatial_size = input_shape[1:3] if backend.image_data_format() == 'channels_last' else input_shape[2:]

        # Compute the kernel and stride sizes according to how large the final feature map will be
        # When the kernel factor and strides are equal, then we can compute the final feature map factor
        # by simply dividing the current factor by the kernel or stride factor
        # The final feature map sizes are 1x1, 2x2, 3x3, and 6x6.
        pool_size = up_size = [spatial_size[0] // level, spatial_size[1] // level]

        x = Pooling2D(pool_size, strides=pool_size, padding='valid', name=pooling_name)(input_tensor)
        x = Conv1x1BnReLU(conv_filters, use_batchnorm, name=conv_block_name)(x)
        # x = layers.UpSampling2D(up_size, interpolation='bilinear', name=upsampling_name)(x)
        x = resize_bilinear(size=(spatial_size[0], spatial_size[1]),align_corners=True)(x)
        return x
    return wrapper

def auxillary_layer( n_filters=256, n_classes=5, dropout=0.1):
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    base_name = "auxillary_layer/"
    def wrapper(input_tensor):
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='auxillary_layer/zeropad1')(input_tensor)
        x = layers.Conv2D(
                filters=n_filters,
                kernel_size=(3, 3),
                padding='valid',
                use_bias = False,
                name='auxillary_layer/conv1',
            )(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        name=base_name + 'bn1')(x)
        x = layers.Activation('relu', name=base_name + 'relu1')(x)
        x = layers.Dropout(rate=dropout)(x)
        x = layers.Conv2D(
                filters=n_classes,
                kernel_size=(1, 1),
                padding='same',
                name='auxillary_layer/conv_out',
            )(x)
        return x 
    return wrapper

# ---------------------------------------------------------------------
#  PSP Decoder
# ---------------------------------------------------------------------

def build_psp(
        backbone_model,
        pooling_type='avg',
        conv_filters=512,
        use_batchnorm=True,
        final_upsampling_factor=8,
        classes=21,
        activation='softmax',
        dropout=None,
        training=True,
        input_shape=None
):
    input_ = backbone_model.input
    x = backbone_model.get_layer(backbone_utils.get_feature_layer()).output
   
    # build spatial pyramid
    x1 = SpatialContextBlock(1, conv_filters, pooling_type, use_batchnorm)(x)
    x2 = SpatialContextBlock(2, conv_filters, pooling_type, use_batchnorm)(x)
    x3 = SpatialContextBlock(3, conv_filters, pooling_type, use_batchnorm)(x)
    x6 = SpatialContextBlock(6, conv_filters, pooling_type, use_batchnorm)(x)

    # aggregate spatial pyramid
    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.Concatenate(axis=concat_axis, name='psp_concat')([x, x1, x2, x3, x6])
    # x = Conv1x1BnReLU(conv_filters, use_batchnorm, name='aggregation')(x)

    # # model regularization
    # if dropout is not None:
    #     x = layers.SpatialDropout2D(dropout, name='spatial_dropout')(x)

    # # model head
    # x = layers.Conv2D(
    #         filters=classes,
    #         kernel_size=(3, 3),
    #         padding='same',
    #         kernel_initializer='glorot_uniform',
    #         name='final_conv',
    #     )(x)
    
    x = cls(conv_filters, use_batchnorm,classes,dropout=dropout)(x)
    print(' input shape is ', input_shape)

    # x = layers.UpSampling2D(final_upsampling_factor, name='final_upsampling', interpolation='bilinear')(x)
    x = resize_bilinear(size=(input_shape[0], input_shape[1]), align_corners=True, name='final_upsampling')(x)
    out = layers.Softmax(name=activation+"_out")(x)

    if(training):     
        aux_l = backbone_model.get_layer(backbone_utils.get_auxillary_layer()).output
        aux_l = auxillary_layer(n_classes=classes)(aux_l)
        aux_l = resize_bilinear(size=(input_shape[0], input_shape[1]), align_corners=True, name='final_upsampling')(aux_l)
        # aux_l = layers.UpSampling2D(final_upsampling_factor, name='final_upsampling_auxillary', interpolation='bilinear')(aux_l)
        
    model = models.Model(input_, [out, aux_l])

    return model


# ---------------------------------------------------------------------
#  PSP Model
# ---------------------------------------------------------------------

def PSPNet(
        backbone_name='vgg16',
        input_shape=(384, 384, 3),
        classes=21,
        activation='softmax',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        psp_conv_filters=512,
        psp_pooling_type='avg',
        psp_use_batchnorm=True,
        psp_dropout=None,
        zoom_factor=8,
        training=True,
        **kwargs
):
    """PSPNet_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model used as feature
                extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``.
            ``H`` and ``W`` should be divisible by ``6 * downsample_factor`` and **NOT** ``None``!
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
                (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        psp_conv_filters: number of filters in ``Conv2D`` layer in each PSP block.
        psp_pooling_type: one of 'avg', 'max'. PSP block pooling type (maximum or average).
        psp_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.
        psp_dropout: dropout rate between 0 and 1.

    Returns:
        ``keras.models.Model``: **PSPNet**

    .. _PSPNet:
        https://arxiv.org/pdf/1612.01105.pdf

    """
    global backbone_utils
    backbone_utils = BackboneUtils(backbone_name)
    # control image input shape
    check_input_shape(input_shape, zoom_factor)

    if(backbone_name == 'resnet50'):
        backbone_model =  DRN50(aux=False, weights=weights, include_top=False, input_shape=input_shape, pooling='avg')
    elif(backbone_name == 'resnet101'):
        backbone_model =  DRN101(aux=False, weights=weights, include_top=False, input_shape=input_shape)

    
    model = build_psp(
        backbone_model,
        pooling_type=psp_pooling_type,
        conv_filters=psp_conv_filters,
        use_batchnorm=psp_use_batchnorm,
        final_upsampling_factor=zoom_factor,
        classes=classes,
        activation=activation,
        dropout=psp_dropout,
        input_shape=input_shape,
        training=training
    )
#    print(model.summary())
#    # lock encoder weights for fine-tuning
#    if encoder_freeze:
#        freeze_model(backbone_model, **kwargs)
#
#    # loading model weights
#    if weights is not None:
#        model.load_weights(weights)

    return model, ['softmax_out', 'lambda_5']
    