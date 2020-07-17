from tensorflow.keras import layers
import tensorflow as tf 
from tensorflow.python.keras.layers import Lambda, multiply;
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend
from tensorflow.keras.losses import Loss, Reduction
import numpy as np
import os
import io
from PIL import Image
from tensorflow.keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer
import cv2
from dataset.DataGeneratorCulane import gray_to_onehot_all
def freeze_model(model):
    """Set all layers non trainable, excluding BatchNormalization layers"""
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return

def resize_bilinear(size=None, align_corners=False, name=None):
    def wrapper(input_tensor):
        return tf.compat.v1.image.resize_bilinear(input_tensor,size=size, align_corners=align_corners, name = name)
    return Lambda(wrapper)

def schedule(init_lr=0.01, power=0.9, max_epochs=100):
    def wrapper(epoch):
        decay = (1 - (epoch / float(max_epochs))) ** power
        res_lr = init_lr * decay
        tf.summary.scalar('learning_rate', data=res_lr, step=epoch)
        return res_lr
    return wrapper

_EPSILON = backend.epsilon()


class IOULOSS(Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self,num_classes, batch_size, input_size, weights,
                 reduction=Reduction.NONE,
                 name='IOULOSS', usesoftmax=True):
        super().__init__(reduction=reduction, name=name)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.reduction = reduction
        self.weighted_loss_fn = weighted_categorical_crossentropy(weights)
        self.usesoftmax = usesoftmax
    
    # @tf.function
    def call(self, y_true, y_pred):
        if(self.usesoftmax):
            y_pred= tf.keras.activations.softmax(y_pred)
        y_pred = backend.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        """ here all calculations will be based on the class greater than 0, except accuracy"""
        avgIOU = backend.variable(0.0)
        
        for i in range(self.batch_size):
           numUnion = backend.variable(1.0)
           recall =  backend.variable(0.0)
           numClass =  backend.variable(0.0)
           IOU = backend.variable(0.0)
           mask = backend.argmax(y_true[i], -1)
           pred = backend.argmax(y_pred[i], -1)
           
           for c in np.arange(1, self.num_classes, 1):
               msk_equal = backend.cast(backend.equal(mask, c),dtype='float32')
               
               masks_sum = backend.sum(msk_equal)
                
               
               predictions_sum = backend.sum(backend.cast(backend.equal(pred,c), 'float32'))

               numTrue = backend.sum(backend.cast(backend.equal(pred,c), 'float32') * backend.cast(backend.equal(mask,c), 'float32'))
               unionSize = masks_sum + predictions_sum - numTrue
               maskhaslabel = backend.greater(masks_sum,0)
               predhaslabel = backend.greater(predictions_sum,0)
               
               predormaskexistlabel = backend.any(backend.stack([maskhaslabel, predhaslabel], axis=0), axis=0)
               
               IOU = backend.switch( predormaskexistlabel , lambda: IOU + numTrue/ unionSize, lambda: IOU)
               numUnion = backend.switch(predormaskexistlabel, lambda: numUnion + 1, lambda:numUnion)
               recall = backend.switch(maskhaslabel ,lambda: recall + numTrue/masks_sum, lambda:recall)
               numClass = backend.switch(maskhaslabel ,lambda: numClass + 1, lambda:numClass)
           IOU= IOU / numUnion
           avgIOU = avgIOU + IOU
        avgIOU = avgIOU / self.batch_size
        iou_loss = 1.0 - avgIOU 
        main_loss = backend.mean(self.weighted_loss_fn(y_true, y_pred))
        return iou_loss + main_loss


def IOULOSSfunc(num_classes, batch_size, input_size, weights, usesoftmax=True):

    weighted_loss_fn = weighted_categorical_crossentropy(weights)
    
    @tf.function
    def wrapper(y_true, y_pred):
        if(usesoftmax):
            y_pred= tf.keras.activations.softmax(y_pred)
        y_pred = backend.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        """ here all calculations will be based on the class greater than 0, except accuracy"""
        avgIOU = 0.0
        
        for i in range(batch_size):
            numUnion = 1.0
            recall =  0.0
            numClass =  0.0
            IOU = 0.0
            mask = backend.argmax(y_true[i], -1)
            pred = backend.argmax(y_pred[i], -1)
           
            for c in np.arange(1, num_classes, 1):
                msk_equal = backend.cast(backend.equal(mask, c),dtype='float32')
               
                masks_sum = backend.sum(msk_equal)
                
               
                predictions_sum = backend.sum(backend.cast(backend.equal(pred,c), 'float32'))

                numTrue = backend.sum(backend.cast(backend.equal(pred,c), 'float32') * backend.cast(backend.equal(mask,c), 'float32'))
                unionSize = masks_sum + predictions_sum - numTrue
                maskhaslabel = backend.greater(masks_sum,0)
                predhaslabel = backend.greater(predictions_sum,0)
               
                predormaskexistlabel = backend.any(backend.stack([maskhaslabel, predhaslabel], axis=0), axis=0)
               
                IOU = backend.switch( predormaskexistlabel , lambda: IOU + numTrue/ unionSize, lambda: IOU)
                numUnion = backend.switch(predormaskexistlabel, lambda: numUnion + 1, lambda:numUnion)
                recall = backend.switch(maskhaslabel ,lambda: recall + numTrue/masks_sum, lambda:recall)
                numClass = backend.switch(maskhaslabel ,lambda: numClass + 1, lambda:numClass)
            IOU= IOU / numUnion
            avgIOU = avgIOU + IOU
        avgIOU = avgIOU / batch_size
        iou_loss = 1.0 - avgIOU 
        # print(np.shape(y_true), np.shape(y_pred))
        main_loss = backend.mean(weighted_loss_fn(y_true, y_pred))
        # dice_loss = soft_dice_loss(y_true, y_pred)
        return  main_loss  + 0.1  * iou_loss
    return wrapper
@tf.function   
def soft_dice_loss(y_true, y_pred):
    # y_true, y_pred = get_valid_labels_and_logits(y_true, y_pred)
    # Next tensors are of shape (num_batches, num_classes)
    tmp = backend.sum(y_true * y_pred, axis=1)
    print('\n\n\n\n\n\n-------------------tmp',np.shape(tmp))
    interception_volume = backend.sum(tmp, axis=1)
    print(np.shape(interception_volume))
    y_true_sum_per_class = backend.sum(backend.sum(y_true, axis=1), axis=1)
    y_pred_sum_per_class = backend.sum(backend.sum(y_pred, axis=1), axis=1)

    return backend.mean(1.0 - 2.0 * interception_volume / (y_true_sum_per_class + y_pred_sum_per_class))
# @tf.function   
# def get_valid_labels_and_logits(y_true, y_pred):
#     valid_labels_mask = tf.not_equal(y_true, 255.0)
#     indices_to_keep = tf.where(valid_labels_mask)
#     valid_labels = tf.gather_nd(params=y_true, indices=indices_to_keep)
#     valid_logits = tf.gather_nd(params=y_pred, indices=indices_to_keep)

#     return valid_labels, valid_logits


@tf.function    
def l2_loss(x1, x2):
    """
    this is Ld which is L2 loss in our case
    """
    loss =  tf.reduce_mean(tf.math.squared_difference(x1,x2))
    return loss

@tf.function
def distill_loss(layer1,layer2, batch_size, shape=[288, 800]):
    
    #print(layer1.get_shape().as_list(), layer2.get_shape().as_list())
    
    # layer1 = tf.expand_dims(layer1, -1)
    # layer2 = tf.expand_dims(layer2, -1)
    #print(layer1.get_shape().as_list(), layer2.get_shape().as_list())
    
    
    # layer1 = tf.image.resize_images(layer1, shape, method=tf.image.ResizeMethod.BILINEAR)
    # layer2 = tf.image.resize_images(layer2, shape, method=tf.image.ResizeMethod.BILINEAR)
    #print(layer1.get_shape().as_list(), layer2.get_shape().as_list())
    
    layer1 = spatial_softmax(layer1,batch_size)
    layer2 = spatial_softmax(layer2,batch_size)
    #print(layer1.get_shape().as_list(), layer2.get_shape().as_list())
    
    distill_loss = l2_loss(layer1, layer2)
    return distill_loss

# function for spatial softmax
@tf.function
def spatial_softmax(x, batch_size=15):
    """
    x is N H W C shaped image, 
    
    """
    print(np.shape(x))
    N, H, W, C = x.get_shape().as_list()
#    print(N, H, W, C, " ---- > shape in sp soft")
    features = tf.reshape(tf.transpose(x, [0,3,1,2]), [batch_size*C, H*W])
    softmax = tf.nn.softmax(features)
    softmax = tf.reshape(softmax, [batch_size, C, H, W])
#    print(np.shape(softmax), 'aftert sotmax reshape')
    softmax = tf.transpose(softmax, [0, 2, 3, 1])
#    print(np.shape(softmax), 'after  softmax, transpose')
    return softmax



def get_iou_loss(num_classes, batch_size, input_size):
    
    def loss(y_true, y_pred):
        
        y_pred = backend.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        print('---=ytrue', np.shape(y_true))
        print('---=ypred', np.shape(y_pred))
        """ here all calculations will be based on the class greater than 0, except accuracy"""
        avgIOU = backend.variable(0.0)
        
        for i in range(batch_size):
           numUnion = backend.variable(1.0)
           recall =  backend.variable(0.0)
           numClass =  backend.variable(0.0)
           IOU = backend.variable(0.0)
           mask = backend.argmax(y_true[i], -1)
           pred = backend.argmax(y_pred[i], -1)
           print('---=mask', np.shape(mask))
           print('---=pred', np.shape(pred))

           mask_shape1 = backend.shape(mask)
           mask_shape1 = backend.print_tensor(mask_shape1, message='mask_shape1 = ' )
           
           for c in np.arange(1, num_classes, 1):
               msk_equal = backend.cast(backend.equal(mask, c),dtype='float32')
               
            #    msks_shape1 = backend.shape(msk_equal)
            #    msks_shape1 = backend.print_tensor(msks_shape1, message='msk_equal shape = ' )
               
               
               masks_sum = backend.sum(msk_equal)
               
               msks_shape2 = backend.shape(masks_sum)
               msks_shape2 = backend.print_tensor(msks_shape2, message='masks_sum shape   = ' )
               
               masks_sum = backend.print_tensor(masks_sum, message='masks_sum = ' )
                
               
               predictions_sum = backend.sum(backend.cast(backend.equal(pred,c), 'float32'))
               
               predictions_sum = backend.print_tensor(predictions_sum, message='predictions_sum = ')
              
               print('---=masks_sum', np.shape(masks_sum))
               print('---=predictions_sum', np.shape(predictions_sum))

               numTrue = backend.sum(backend.cast(backend.equal(pred,c), 'float32') * backend.cast(backend.equal(mask,c), 'float32'))
               unionSize = masks_sum + predictions_sum -numTrue
            #    unionSize = tf.Print(unionSize, [unionSize], "union size : ")
               unionSize = backend.print_tensor(unionSize, message='unionSize = ')
               
               maskhaslabel = backend.greater(masks_sum,0)
               predhaslabel = backend.greater(predictions_sum,0)
               
               maskhaslabel = backend.print_tensor(maskhaslabel, message='maskhaslabel = ')
               predhaslabel = backend.print_tensor(predhaslabel, message='maskhaslabel = ')
               
               predormaskexistlabel = backend.any(backend.stack([maskhaslabel, predhaslabel], axis=0), axis=0)# backend.cond(backend.logical_or(maskhaslabel, predhaslabel), lambda:True,lambda:False)
               predormaskexistlabel = backend.print_tensor(predormaskexistlabel, message='predormaskexistlabel = ')
               
               
               IOU = backend.switch( predormaskexistlabel , lambda: IOU + numTrue/ unionSize, lambda: IOU)
               numUnion = backend.switch(predormaskexistlabel, lambda: numUnion + 1, lambda:numUnion)
               recall = backend.switch(maskhaslabel ,lambda: recall + numTrue/masks_sum, lambda:recall)
               numClass = backend.switch(maskhaslabel ,lambda: numClass + 1, lambda:numClass)
           IOU= IOU / numUnion
           avgIOU = avgIOU + IOU
        avgIOU = avgIOU / batch_size
        iou_loss = 1.0 - avgIOU 
        
        # iou_loss = backend.print_tensor(iou_loss, message='\n\n\niouloss = ')
        return iou_loss          
    return loss


def weighted_categorical_crossentropy(weights, prop=0.1, scale=False):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = backend.variable(weights)        
    @tf.function
    def loss(y_true, y_pred):
        
        # scale predictions so that the class probas of each sample sum to 1
        if(scale):
            y_pred /= backend.sum(y_pred, axis=-1, keepdims=True) 
        # clip to prevent NaN's and Inf's
        y_pred = backend.clip(y_pred, backend.epsilon(), 1 - backend.epsilon())
        # calc 
        loss = y_true * backend.log(y_pred) * weights
        loss = -backend.sum(loss, -1)
        # loss = backend.print_tensor(loss, message='\n\n\niouloss = ')
        # lossnp = loss.numpy()
        
        # lossnp[lossnp<0.01] = 0
        # loss.assign(lossnp)
        
        return loss
    return loss

def iou_categorical_crossentropy(n_classes, batch_size, input_size, weights):
    # weighted_loss_fn = weighted_categorical_crossentropy(weights, 0.00000000000000000001)
    iou_loss_ = get_iou_loss(n_classes, batch_size, input_size)
    def loss(y_true, y_pred):
#        loss_main =weighted_loss_fn(y_true, y_pred)
#        _,_,_, loss_iou = get_iou_loss(y_true, y_pred, n_classes, batch_size, input_size)
        loss_iou = iou_loss_(y_true, y_pred)
        # print('\n\n\n\n\nshape of loss iou', np.shape(loss_iou),'\n\n\n\n\n')
        return loss_iou
    return loss
def categorical_crossentropy_weighted(weights,  prop=0.1):
    weighted_loss_fn = weighted_categorical_crossentropy(weights, prop=prop)
    def loss(y_true, y_pred):
        loss_main =weighted_loss_fn(y_true, y_pred)
        # print('\n\n\n\n\nshape of loss categorical_crossentropy', np.shape(loss_main),'\n\n\n\n\n')
#        _,_,_, loss_iou = get_iou_loss(y_true, y_pred, n_classes, batch_size, input_size)
        return loss_main #+ loss_iou
    return loss


class ModelDiagonoser(Callback):

    def __init__(self,
                 data_generator,
                 batch_size,
                 num_samples,
                 output_dir, input_shape, n_classes):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.tensorboard_writer =  tf.summary.create_file_writer(output_dir+"/diagnose/", flush_millis=10000)
        self.data_generator = data_generator
        self.input_shape = input_shape
        self.colors = np.array([
                [255, 255, 0],
                [255,  0,  0],
                [0, 255,   0],
                [0,   0, 255],
                [0, 0, 0]])
        self.color_dict = {
                      0: (0,0,0),
                      1: (0,255,0)
                      }
        self.n_classes = n_classes
        self.colors = self.colors[:self.n_classes]
        is_sequence = isinstance(self.data_generator, Sequence)
        if is_sequence:
            self.enqueuer = OrderedEnqueuer(self.data_generator,
                                            use_multiprocessing=True,
                                            shuffle=False)
        else:
            self.enqueuer = GeneratorEnqueuer(self.data_generator,
                                              use_multiprocessing=True,
                                              wait_time=0.01)
        self.enqueuer.start(workers=4, max_queue_size=4)
        
        
    def on_epoch_end(self, epoch, logs=None):
        output_generator = self.enqueuer.get()
        generator_output = next(output_generator)
        x, y = generator_output[:2]
        y_pred = self.model.predict(x)
        # y_pred1 = self.model.predict(x)
        y_pred1,y_pred2 = y_pred[0], y_pred[-1]
        y_pred1 = np.argmax(y_pred1, axis=-1)
        y_pred2 = np.argmax(y_pred2, axis=-1)
        
        # y = y1['softmax_out']
        y_imgs = [] 
        y_imgs_pred = [] 
        y_imgs_pred1 = [] 
        x_imgs = []
#        n_classes = np.shape(y)[-1]
        # print(np.unique(np.argmax(y, -1)), np.unique(y_pred1),"\n\n\n\n\n")
        y_pred1 = gray_to_onehot_all(y_pred1,self.color_dict)
        y_pred2 = gray_to_onehot_all(y_pred2,self.color_dict)
        # print(np.shape(y_pred1), np.shape(y_pred2), np.shape(y))
        for i in range(len(y)):
            y_img = np.resize(np.dot(np.reshape(y[i], (-1, self.n_classes)), self.colors) ,self.input_shape)
            y_img_pred = np.resize(np.dot(np.reshape(y_pred1[i], (-1, self.n_classes)), self.colors) ,self.input_shape)
            y_img_pred1 = np.resize(np.dot(np.reshape(y_pred2[i], (-1, self.n_classes)), self.colors) ,self.input_shape)
            
            y_imgs.append(y_img)
            y_imgs_pred.append(y_img_pred)
            y_imgs_pred1.append(y_img_pred1)
            x_imgs.append(x[i].astype('uint8'))
            
        y_imgs = np.array(y_imgs)
        x_imgs = np.array(x_imgs)
        y_imgs_pred = np.array(y_imgs_pred)
        y_imgs_pred1 = np.array(y_imgs_pred1)
        
        

        
        with self.tensorboard_writer.as_default():
            is_written = tf.summary.image("img", x_imgs , step=epoch)
            is_written = tf.summary.image("train/gts", y_imgs, step=epoch)
            is_written = tf.summary.image("train/predictions1", y_imgs_pred, step=epoch)
            is_written = tf.summary.image("train/predictions2", y_imgs_pred1, step=epoch)
            # if(is_written):
                # print(' image has written to the tensorboard')
        self.tensorboard_writer.flush()

    def on_train_end(self, logs=None):
        self.enqueuer.stop()
        self.tensorboard_writer.close()
        
        
def seg_metrics(y_true, y_pred, metric_name,
    metric_type='standard', drop_last = True, mean_per_class=False, verbose=False):
    """
    Compute mean metrics of two segmentation masks, via Keras.

    IoU(A,B) = |A & B| / (| A U B|)
    Dice(A,B) = 2*|A & B| / (|A| + |B|)

    Args:
        y_true: true masks, one-hot encoded.
        y_pred: predicted masks, either softmax outputs, or one-hot encoded.
        metric_name: metric to be computed, either 'iou' or 'dice'.
        metric_type: one of 'standard' (default), 'soft', 'naive'.
          In the standard version, y_pred is one-hot encoded and the mean
          is taken only over classes that are present (in y_true or y_pred).
          The 'soft' version of the metrics are computed without one-hot
          encoding y_pred.
          The 'naive' version return mean metrics where absent classes contribute
          to the class mean as 1.0 (instead of being dropped from the mean).
        drop_last = True: boolean flag to drop last class (usually reserved
          for background class in semantic segmentation)
        mean_per_class = False: return mean along batch axis for each class.
        verbose = False: print intermediate results such as intersection, union
          (as number of pixels).
    Returns:
        IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
          in which case it returns the per-class metric, averaged over the batch.

    Inputs are B*W*H*N tensors, with
        B = batch size,
        W = width,
        H = height,
        N = number of classes
    """

    flag_soft = (metric_type == 'soft')
    flag_naive_mean = (metric_type == 'naive')

    # always assume one or more classes
    num_classes = backend.shape(y_true)[-1]

    if not flag_soft:
        # get one-hot encoded masks from y_pred (true masks should already be one-hot)
        y_pred = backend.one_hot(backend.argmax(y_pred), num_classes)
        y_true = backend.one_hot(backend.argmax(y_true), num_classes)

    # if already one-hot, could have skipped above command
    # keras uses float32 instead of float64, would give error down (but numpy arrays or keras.to_categorical gives float64)
    y_true = backend.cast(y_true, 'float32')
    y_pred = backend.cast(y_pred, 'float32')

    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1,2) # W,H axes of each image
    intersection = backend.sum(backend.abs(y_true * y_pred), axis=axes)
    mask_sum = backend.sum(backend.abs(y_true), axis=axes) + backend.sum(backend.abs(y_pred), axis=axes)
    union = mask_sum  - intersection # or, np.logical_or(y_pred, y_true) for one-hot

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth)/(mask_sum + smooth)

    metric = {'iou': iou, 'dice': dice}[metric_name]

    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask =  backend.cast(backend.not_equal(union, 0), 'float32')

    if drop_last:
        metric = metric[:,:-1]
        mask = mask[:,:-1]

    if verbose:
        print('intersection, union')
        print(backend.eval(intersection), backend.eval(union))
        print(backend.eval(intersection/union))

    # return mean metrics: remaining axes are (batch, classes)
    if flag_naive_mean:
        return backend.mean(metric)

    # take mean only over non-absent classes
    class_count = backend.sum(mask, axis=0)
    non_zero = tf.greater(class_count, 0)
    non_zero_sum = tf.boolean_mask(backend.sum(metric * mask, axis=0), non_zero)
    non_zero_count = tf.boolean_mask(class_count, non_zero)
    
    if verbose:
        print('Counts of inputs with class present, metrics for non-absent classes')
        print(backend.eval(class_count), backend.eval(non_zero_sum / non_zero_count))

    return backend.mean(non_zero_sum / non_zero_count)

def mean_iou(y_true, y_pred, **kwargs):
    """
    Compute mean Intersection over Union of two segmentation masks, via Keras.

    Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return seg_metrics(y_true, y_pred, metric_name='iou', **kwargs)

def mean_dice(y_true, y_pred, **kwargs):
    """
    Compute mean Dice coefficient of two segmentation masks, via Keras.

    Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return seg_metrics(y_true, y_pred, metric_name='dice', **kwargs)