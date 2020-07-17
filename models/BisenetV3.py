# -*- coding: utf-8 -*-
from models import Conv2dBn
from utils import resize_bilinear
from tensorflow.keras import backend
from tensorflow.keras import models 
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
from utils._utils import IOULOSSfunc, distill_loss
wd = 1e-3
def DetailedBranch():
    def wrapper(input_tensor):
        x= Conv2dBn(
            64,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 2,
            use_batchnorm=True, 
            use_bias=False,
            name="detail_branch_cnn_1_1",
            kernel_regularizer=l2(wd),
        )(input_tensor)
        
        x= Conv2dBn(
            64,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            use_bias=False,
            name="detail_branch_cnn_1_2",
            kernel_regularizer=l2(wd)
        )(x)
        #---------------------------------------------------------------
        x= Conv2dBn(
            64,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 2,
            use_batchnorm=True,
            use_bias=False,
            name="detail_branch_cnn_2_1",
            kernel_regularizer=l2(wd)
        )(x)
        
        x= Conv2dBn(
            64,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            use_bias=False,
            name="detail_branch_cnn_2_2",
            kernel_regularizer=l2(wd)
        )(x)
        x= Conv2dBn(
            64,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            use_bias=False,
            name="detail_branch_cnn_2_3",
            kernel_regularizer=l2(wd)
        )(x)
        #---------------------------------------------------------------
        x= Conv2dBn(
            128,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 2,
            use_batchnorm=True,
            name="detail_branch_cnn_3_1",
        )(x)
        
        x= Conv2dBn(
            128,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            use_bias=False,
            name="detail_branch_cnn_3_2",
            kernel_regularizer=l2(wd)
        )(x)
        x= Conv2dBn(
            128,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            use_bias=False,
            name="detail_branch_cnn_3_3",
            kernel_regularizer=l2(wd)
        )(x)
        return x
    return wrapper

def StemBlock(out_channels=16):
    def wrapper(input_tensor):
        x= Conv2dBn(
            out_channels,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 2,
            use_batchnorm=True,
            use_bias=False,
            name="stem_block_cnn",
            kernel_regularizer=l2(wd)
        )(input_tensor)
        #----------------------------
        x1= Conv2dBn(
            out_channels/2,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            use_bias=False,
            name="stem_block_cnndown_cnn1",
            kernel_regularizer=l2(wd)
        )(x)
        x1= Conv2dBn(
            out_channels,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 2,
            use_batchnorm=True,
            use_bias=False,
            name="stem_block_cnndown_cnn3",
            kernel_regularizer=l2(wd)
        )(x1)
        #---------------------------------
        #--------maxpooling--------------
        x2 = layers.MaxPool2D(
                pool_size=(3, 3), strides=2, padding='same',name="stem_block_maxpool"
            )(x)
        # x2, pooling_indices = tf.nn.max_pool_with_argmax(x1,
        #                 ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        
        # concatenate 
        x = layers.Concatenate(axis=-1)([x1, x2])
        x= Conv2dBn(
            out_channels,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            use_bias=False,
            name="stem_block_cnn_out",
            kernel_regularizer=l2(wd)
        )(x)
        return x
    return wrapper
def GEBlock(stride=1, out_channels=16, e=6, name=""):
    """
    Gather and expansion layer
    Arguments:
        out channel
        input tensor
        stride
    """
    def wrapper(input_tensor):
        x= Conv2dBn(
            out_channels,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            use_bias=False,
            name=name+"ge_block_cnn3_1",
            kernel_regularizer=l2(wd)
        )(input_tensor)
        x = layers.DepthwiseConv2D(
            3, strides=(stride, stride), padding='same',depth_multiplier=e, use_bias=False, name=name+"geblock_dwconv3_1"
        )(x)
        if(stride==2):
            x = layers.DepthwiseConv2D(
                3, strides=(1, 1), padding='same', use_bias=False, name=name+"geblock_dwconv3_2", kernel_regularizer=l2(wd)
            )(x)    
        x= Conv2dBn(
            out_channels,
            kernel_size=1,
            activation=None,
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            use_bias=False,
            name=name+"ge_block_cnn1_1",
            kernel_regularizer=l2(wd)
        )(x)
        if(stride==2):
            x1 = layers.DepthwiseConv2D(
                3, strides=(stride, stride), padding='same', use_bias=False, name=name+"geblock_dwconv3_3_shortcut"
            )(input_tensor) 
            x1 = Conv2dBn(
                out_channels,
                kernel_size=1,
                activation=None,
                kernel_initializer='he_uniform',
                padding='same',
                strides = 1,
                use_batchnorm=True,
                name=name+"ge_block_cnn1_2_shortcut",
                use_bias=False,
                kernel_regularizer=l2(wd)
            )(x1)
            x = layers.Add()([x, x1])    
        else:
            x = layers.Add()([x, input_tensor])
        return x
    return wrapper

def ContextEmpedding(out_channels=16, stride=1, name=""):
    def wrapper(input_tensor):
        x = backend.mean(input_tensor, axis=[1,2], keepdims=True) #layers.GlobalAveragePooling2D(axis=-1)(input_tensor)
        # print(1, np.shape(x))
        x = Conv2dBn(
            out_channels,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            name=name+"ce_block_cnn1_1",
            use_bias=False,
            kernel_regularizer=l2(wd)
            
        )(x)
        # print(2, np.shape(x))
        x = layers.Add()([x, input_tensor])
        # print(3, np.shape(x))
        x= Conv2dBn(
            out_channels,
            kernel_size=3,
            activation=None,
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=False,
            use_bias=False,
            name=name+"ge_block_cnn3_1",
            kernel_regularizer=l2(wd)
        )(x)
        # print(4, np.shape(x))
        return x
    return wrapper

def AggregationLayer(name="", out_channels=16):
    def wrapper(detailed_in, semantic_in):
        #--------------preparing detailed_in----------------------------------
        _, w, h, _ = np.shape(detailed_in)
        x_d = layers.DepthwiseConv2D(
            3, strides=(1, 1), padding='same',depth_multiplier=1, use_bias=False, name=name+"agglayer_dwconv1_1"
        )(detailed_in)
        x_d = Conv2dBn(
            out_channels,
            kernel_size=1,
            activation=None,
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=False,
            name=name+"agglayer_cnn1_1",
            use_bias=False,
            kernel_regularizer=l2(wd)
        )(x_d)
        # now prepare detailed in to be the mixed with semantic
        x_d2s= Conv2dBn(
            out_channels,
            kernel_size=3,
            activation=None,
            kernel_initializer='he_uniform',
            padding='same',
            strides = 2,
            use_batchnorm=True,
            use_bias=False,
            name=name+"agglayer_cnn3_1",
            kernel_regularizer=l2(wd)
        )(detailed_in)
        x_d2s = layers.AveragePooling2D(
            pool_size=(3, 3), strides=(2,2), padding='same'
        )(x_d2s)
        #------------------------prepare semantic in--------------------------
        x_s = layers.DepthwiseConv2D(
            3, strides=(1, 1), padding='same',depth_multiplier=1, use_bias=False, name=name+"agglayer_dwconv1_2"
        )(semantic_in)
        # print(5, np.shape(x_s))
        x_s = Conv2dBn(
            out_channels,
            kernel_size=1,
            activation=None,
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=False,
            name=name+"agglayer_cnn1_2",
            use_bias=False,
            kernel_regularizer=l2(wd)
        )(x_s)
        #now prepare semantic in to be mixed with detailed
        x_s2d = Conv2dBn(
            out_channels,
            kernel_size=3,
            activation=None,
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            use_bias=False,
            name=name+"agglayer_cnn3_2",
            kernel_regularizer=l2(wd)
        )(semantic_in)
        x_s2d = resize_bilinear(size=(w,h),name=name+'resize_bil_1')(x_s2d)
        # x_s2d = layers.UpSampling2D(
            # size=(4, 4),  interpolation='bilinear'
        # )(x_s2d)
        # mixing-----------------------------------------------
        x_d = layers.Multiply()([x_d , x_s2d])
        x_s = layers.Multiply()([x_s , x_d2s])
        # x_s = layers.UpSampling2D(
            # size=(4, 4),  interpolation='bilinear'
        # )(x_s)
        x_s = resize_bilinear(size=(w,h),name=name+'resize_bil_1')(x_s)
        x = layers.Add()([x_s, x_d])
        
        x = Conv2dBn(
            out_channels,
            kernel_size=3,
            activation=None,
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=True,
            use_bias=False,
            name=name+"agglayer_output",
            kernel_regularizer=l2(wd)
        )(x)
        return x
    return wrapper

def SemanticBranch(channels= 16):
    def wrapper(input_tensor):
        stm = StemBlock(out_channels=channels)(input_tensor)
        ge1 = GEBlock(stride=2,out_channels=channels*2, e=6,name='GE1_1')(stm)
        ge1 = GEBlock(stride=1,out_channels=channels*2, e=6,name='GE1_2')(ge1)
        
        ge2 = GEBlock(stride=2,out_channels=channels*4, e=6,name='GE2_1')(ge1)
        ge2 = GEBlock(stride=1,out_channels=channels*4, e=6,name='GE2_2')(ge2)
        
        ge3 = GEBlock(stride=2,out_channels=channels*8, e=6,name='GE3_1')(ge2)        
        ge3 = GEBlock(stride=1,out_channels=channels*8, e=6,name='GE3_2')(ge3)
        ge3 = GEBlock(stride=1,out_channels=channels*8, e=6,name='GE3_3')(ge3)
        ge3 = GEBlock(stride=1,out_channels=channels*8, e=6,name='GE3_4')(ge3)
        
        ce = ContextEmpedding(out_channels=channels*8, stride=1,name="ceblock")(ge3)
        return stm, ge1, ge2,ge3,ce
    return wrapper
def HeadBlock(classes, channels, h, w ,name ="", only_resize=False):
    
    def wrapper(x):
        if(not only_resize):
            # segmentation head
            x = Conv2dBn(
                channels * 8,
                kernel_size=3,
                activation='relu',
                kernel_initializer='he_uniform',
                padding='same',
                strides = 1,
                use_batchnorm=True,
                use_bias=False,
                name=name + 'seghead_con3_1',
                kernel_regularizer=l2(wd)
            )(x)
            
        x = Conv2dBn(
            classes,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            strides = 1,
            use_batchnorm=False,
            use_bias=False,
            name=name + 'seghead_conv_end',
            kernel_regularizer=l2(wd)
        )(x)
        
        # if(not self.scale == 1):
            # x = layers.UpSampling2D(
            #     size=(self.scale, self.scale),  interpolation='bilinear'
            # )(x)
        x = resize_bilinear(size=(h, w),name=name + 'resize_bil_end')(x)
        return x
    return wrapper
# class BisenetV2:
#     def __init__(self,  input_shape=(384, 384, 3), classes = 5, training=True):
#         self.h =input_shape[0]
#         self.w= input_shape[1]
#         self.classes= classes
#         # self.scale=scale
#         self.training=training
#         self.init_model()
        
        
        
        
#     def init_model(self):
        
#         self.input = layers.Input(shape=[self.h,self.w,3])
#         self.y_true = layers.Input(shape=[self.h,self.w,self.classes])
#         xd = DetailedBranch()(self.input)
#         channels = 8 # based on these channels you can change the complexity of the model
#         stm, ge1,ge2,ge3, xs = SemanticBranch(channels)(self.input)
#         ae = AggregationLayer(out_channels=channels * 8)(xd, xs)
#         self.logits = HeadBlock(self.classes, channels, self.h, self.w)(ae)
#         self.logits_stm = HeadBlock(self.classes, channels-8, self.h, self.w, name="stm_aux_",only_resize=True)(stm)
#         self.logits_ge1 = HeadBlock(self.classes, channels-6, self.h, self.w, name="ge1_aux_",only_resize=True)(ge1)
#         self.logits_ge2 = HeadBlock(self.classes, channels-4, self.h, self.w, name="ge2_aux_",only_resize=True)(ge2)
#         self.logits_ge3 = HeadBlock(self.classes, channels-2, self.h, self.w, name="ge3_aux_",only_resize=True)(ge3)
        
        
#         model = models.Model(inputs=[self.input, self.y_true], outputs=[self.logits])
        
#         return model


class BisenetV2Model:
    
    def __init__(self, train_op, input_shape=(384, 384, 1), classes = 5, training=True, batch_size=15, class_weights=[0.001,1]):
        # xavier=tf.keras.initializers.GlorotUniform()
        self.h =input_shape[0]
        self.w= input_shape[1]
        self.classes= classes
        # self.scale=scale
        self.training=training
        
        
        self.channels = 16 # based on these channels you can change the complexity of the model
        self.batch_size = batch_size        
        self.main_loss = IOULOSSfunc(self.classes, self.batch_size , input_shape, class_weights)
        
        # model
        self.input = layers.Input(shape=[self.h,self.w,input_shape[-1]], dtype=tf.float32)
        self.detailed_branch = DetailedBranch()
        self.semantic_branc = SemanticBranch(self.channels)
        self.aggregation_layer = AggregationLayer(out_channels=self.channels * 8)
        self.head_block_logits_end  = HeadBlock(self.classes, self.channels , self.h, self.w, name="logits_end")
        self.head_block_logits_stm = HeadBlock(self.classes, self.channels -8, self.h, self.w, name="stm_aux_",only_resize=True)
        self.head_block_logits_ge1 = HeadBlock(self.classes, self.channels -6, self.h, self.w, name="ge1_aux_",only_resize=True)
        self.head_block_logits_ge2 = HeadBlock(self.classes, self.channels -4, self.h, self.w, name="ge2_aux_",only_resize=True)
        self.head_block_logits_ge3 = HeadBlock(self.classes, self.channels -2, self.h, self.w, name="ge3_aux_",only_resize=True)
        xd = self.detailed_branch(self.input)       
        stm, ge1,ge2,ge3, xs =self.semantic_branc(self.input)
        ae = self.aggregation_layer(xd, xs)
        print('before upsampling \n\n\n\n', np.shape(ae))
        # logits = self.head_block_logits1(ae)
        # logits = self.head_block_logits2(logits)
        logits = self.head_block_logits_end(ae)
        print('after upsampling \n\n\n\n', np.shape(logits))
        logits_stm = self.head_block_logits_stm(stm)
        logits_ge1 = self.head_block_logits_ge1(ge1)
        logits_ge2 = self.head_block_logits_ge2(ge2)
        logits_ge3 = self.head_block_logits_ge3(ge3)       
        self.model = models.Model(self.input, [logits, logits_stm, logits_ge1,logits_ge2,logits_ge3])
        self.train_op = train_op
      
    #Custom loss fucntion
    def get_loss(self,X,Y):
        
        y_prob, stm_prob, ge1_prob, ge2_prob, ge3_prob = self.model(X)
        
        aux_loss1  = distill_loss(stm_prob, ge1_prob, self.batch_size)
        aux_loss2  = distill_loss(ge1_prob, ge2_prob, self.batch_size)
        aux_loss3  = distill_loss(ge2_prob, ge3_prob, self.batch_size)
        aux_loss = tf.reduce_sum([aux_loss1, aux_loss2, aux_loss3])
        main_loss = self.main_loss(Y, y_prob)
        return main_loss, y_prob  # + 0.1 * aux_loss, y_prob #  + 0.1 * aux_loss
      
    # get gradients
    def get_grad(self,X,Y):
        with tf.GradientTape() as tape:
            L,y_pred = self.get_loss(X,Y)
            g = tape.gradient(L, self.model.trainable_variables)
        return g, y_pred, L
      
    # perform gradient descent
    def network_learn(self,X,Y):
        g, y_pred, L = self.get_grad(X,Y)
        self.train_op.apply_gradients(zip(g, self.model.trainable_variables))
        return y_pred, L