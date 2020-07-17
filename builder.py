import os 
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow.keras.optimizers  as optimizers
from models import PSPNet,  BisenetV2Model
import tensorflow.keras as keras
from utils import schedule
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras import backend
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
import numpy as np 
import tensorflow as tf
from dataset import DataGeneratorCulane, transforms
from tensorflow.python.keras.layers import Lambda
from tensorflow.keras.utils import multi_gpu_model
import cv2
from utils._utils import  ModelDiagonoser, categorical_crossentropy_weighted,  mean_iou, IOULOSS
import shutil
import pickle
# backend.clear_session()
# tf.compat.v1.disable_eager_execution()
# tf.config.experimental_run_functions_eagerly(True)
#from numba import cuda
#cuda.select_device(0)
#cuda.close()
# tf.enable_v2_behavior()
# tf.enable_eager_execution()

class Builder():
    def __init__(self, model_name="Bisenet_V2", optimizer_name='Adam',loss_names=['iou_categorical_crossentropy','iou_categorical_crossentropy'],
    metrics_names=['accuracy'], training=True, transfer_learning=False, input_shape=(288,800,3), n_classes=2):
        """
        -----------------------------------------------------------------------
        1. losses
            1. categorical_crossentropy
            2. iou_categorical_crossentropy
            3. categorical_crossentropy_weighted
        2. models
            1. PSPNet
            2. Bisenet_V2
        3. optimizers
            1. Adam
            2. SGD
        4. metrics
            1. accuracy
        5. lr_decay_policy
            1. poly
        -----------------------------------------------------------------------
        """
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.training=training
        self.n_classes = n_classes
        self.loss_names =loss_names
        self.metrics_names = metrics_names
        self.momentum = 0.9
        self.weight_decay = 0.9
        self.input_shape=input_shape
        self.lr = [0.00004, 0.0001, 0.0005, 0.001, 0.00006][4]
        self.epochs = 200
        self.lr_decay_policy='poly'
        self.lr_decay_power = 0.9
        self.batch_size = 32
        self.class_weights=[[0.0025,1.0],[0.0025,1.0], [0.0025,1.0], [0.0025,1.0], [0.2,1.0]][4]#, 1.0,1.0,1.0]
        self.backbone='resnet50'
        self.model_out_layer_names = None
        self.pretrained_encoder_weights='imagenet'
         # learning decay will be applied in each #no epochs and the shape will look like staircase
        self.lr_decay_staircase=True
        self.transfer_learning = transfer_learning
        self.train_id ='bisnet_v2_rgb'
        self.logdir = "/home/essys/projects/segmentation/logs/"+self.train_id +"/"
        

        self.init()

    def init(self,):
#        strategy = tf.distribute.MirroredStrategy()
#        BATCH_SIZE = self.batch_size * strategy.num_replicas_in_sync
#        with strategy.scope():
        print('init model\n\n\n\n')
        self.init_model()
        print('init optimizer\n\n\n\n')
        
        if(not self.model_name=="Bisenet_V2"):
            self.init_optimizer()
            print('init losses\n\n\n\n')        
            self.init_losses()
            print('init metrices\n\n\n\n')
            self.init_metrics()
            print('init weight decay\n\n\n\n')
    #        self.add_weight_decay()
            print('init callbacks\n\n\n\n')
        
        
        
        project_dir = "./"
        
        
        data_dir = project_dir + "data/"
        
        train_img_paths = np.array(pickle.load(open(data_dir + "train_img_paths.pkl", 'rb')))
        train_img_paths = train_img_paths
        self.n_samples = len(train_img_paths)
        train_trainId_label_paths = np.array(pickle.load(open(data_dir + "train_trainId_label_paths.pkl", 'rb')))
        train_trainId_label_paths = train_trainId_label_paths
#        train_existance_labels = np.array(pickle.load(open(data_dir + "train_existance_label.pkl", 'rb')))
        self.train_mean_channels = pickle.load(open("data/mean_channels.pkl", 'rb'))
        input_mean =  self.train_mean_channels #[103.939, 116.779, 123.68] # [0, 0, 0]
        input_std = [1, 1, 1]
        ignore_label = 255
        augmenters = []
        augmenters_val = []
        #scaler = transforms.GroupRandomScale(size=(0.595, 0.621), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST))
        cropper = transforms.GroupRandomCropKeepBottom(cropsize=(400, 250))    
        rotater= transforms.GroupRandomRotation(degree=(-15, 15), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=(input_mean, (0, )))
        normalizer = transforms.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, )))
        color_augmenter = transforms.GroupColorAugment()
        eraser = transforms.GroupRandomErase(mean=input_mean)
         
             
        # add augmenters by their order for train
        #    augmenters.append(scaler)
        #augmenters.append(color_augmenter)
        augmenters.append(eraser)
        augmenters.append(rotater)
        augmenters.append(cropper)
        augmenters.append(normalizer)

        self.training_generator = DataGeneratorCulane.DataIterators(train_img_paths,train_trainId_label_paths, batch_size=self.batch_size, 
                                                                dim=(self.input_shape[0], self.input_shape[1]),n_classes=self.n_classes, 
                                                                n_channels=1, output_of_models=(self.model_out_layer_names if self.model_name =='PSPNet' else None), transform=augmenters)
        if(not self.model_name=="Bisenet_V2"):
            self.init_callbacks()
            print('compile model\n\n\n\n')
    
            # try:
    #        self.model = multi_gpu_model(self.model, gpus=4)
                # print("Training using multiple GPUs..")
            # except:
                # print("Training using single GPU or CPU..")
        if(self.model_name=="Bisenet_V2"):
            # weighted_loss_fn = weighted_categorical_crossentropy(self.class_weights)
            # #y_true,  y_pred, y_stm, y_ge1,y_ge2, y_ge3, num_classes, batch_size, input_size, weights,weighted_loss_fn, usesoftmax=True):        
            # self.loss_bisenetv2 = IOULOSS_BISENETV2_func(self.model_base.y_true, self.model_base.logits, self.model_base.logits_stm, 
            #                               self.model_base.logits_ge1,self.model_base.logits_ge2,self.model_base.logits_ge3,
            #                               self.n_classes,self.batch_size,self.input_shape,self.class_weights,
            #                               weighted_loss_fn,True)
            # self.model.add_loss(self.loss_bisenetv2)
            # self.model.compile(optimizer=self.optimizer, metrics=self.metrics)    
            pass
        else:
            self.model.compile(loss=self.losses, optimizer=self.optimizer, metrics=self.metrics, run_eagerly=True)

    def demo_train(self):
        
        len_steps = self.n_samples/self.batch_size #66785/self.batch_size
        if(self.model_name == "Bisenet_V2"):  
            iterator = self.training_generator.get_data_iterator()                
            tensorboard_writer =  tf.summary.create_file_writer(self.logdir+"/diagnose/", flush_millis=10000)
            color_dict = {
                      0: (0,0,0),
                      1: (0,255,0)
                      }
            colors = np.array([
                [0,  0,  0],
                [0, 255, 0]
                ])
            
            for epoch in range(22, self.epochs):
                for i in range(int(len_steps)):
                    
                    step = int(epoch*len_steps) + i
                    x, y = next(iterator)      
                    print(np.shape(x), np.shape(y))
                    if(len(x) <self.batch_size): continue
                    y_pred, loss = self.model.network_learn(x,y)                    
                    
                    print("Step {}, Loss: {}".format(self.model.train_op.iterations.numpy(), loss.numpy()))
                    if(i % 20 == 0):
                        m_iou = mean_iou(y,y_pred).numpy()
                        
                        
                        
                        y_pred = np.argmax(y_pred, axis=-1)
                        _y = np.argmax(y, axis=-1)
                        
                        accuracy = backend.sum(backend.cast(backend.equal(_y,y_pred), tf.float32)) /(self.batch_size * self.input_shape[0] * self.input_shape[1]) 
                        
                        # y = y1['softmax_out']
                        y_imgs = [] 
                        y_imgs_pred = [] 
                        x_imgs = []
                #        n_classes = np.shape(y)[-1]
                        # print(np.unique(np.argmax(y, -1)), np.unique(y_pred1),"\n\n\n\n\n")
                        y_pred = DataGeneratorCulane.gray_to_onehot_all(y_pred, color_dict)
                        
                        
                        for i in range(len(y)):
                            y_img = np.resize(np.dot(np.reshape(y[i], (-1, self.n_classes)), colors) ,self.input_shape)
                            y_img_pred = np.resize(np.dot(np.reshape(y_pred[i], (-1, self.n_classes)), colors) ,self.input_shape)                        
                            y_imgs.append(y_img)
                            y_imgs_pred.append(y_img_pred)                    
                            x_imgs.append((x[i] + self.train_mean_channels).astype('uint8'))
                            
                        y_imgs = np.array(y_imgs,dtype=np.uint8)
                        x_imgs = np.array(x_imgs)
                        y_imgs_pred = np.array(y_imgs_pred)
                        
                        with tensorboard_writer.as_default():
                            is_written = tf.summary.image("img", x_imgs , step=step)
                            is_written = tf.summary.image("train/gts", y_imgs, step=step)
                            is_written = tf.summary.image("train/predictions1", y_imgs_pred, step=step)
                            tf.summary.scalar("miou", m_iou, step=step)
                            tf.summary.scalar("learning_rate", self.model.train_op.learning_rate.numpy(), step=step)
                            tf.summary.scalar("loss", loss.numpy(), step=step)
                            tf.summary.scalar("accuracy", accuracy.numpy(), step=step)
                            if(is_written):
                                print(' image has written to the tensorboard')
                        tensorboard_writer.flush()
                    if((step+1) % 500 == 0):
                        checkpoint_path = '/home/essys/projects/segmentation/checkpoints/' + self.train_id + "/" +"cp-epoch-{}-step-{}.ckpt".format(epoch, step)
                        print(checkpoint_path)
                        self.model.model.save_weights(checkpoint_path)
                    else:
                        print(step)                    
                new_learning_rate = self.model.train_op.learning_rate.numpy() * (1-(self.model.train_op.learning_rate.numpy()/self.epochs)) ** 2800
                
                backend.set_value(self.model.train_op.learning_rate, new_learning_rate)
                    
                    
            tensorboard_writer.close()
        else:
            self.model.fit_generator(self.training_generator, steps_per_epoch=len_steps, 
                                      epochs=self.epochs, callbacks=self.callbacks)
        
    def init_callbacks(self,):
        scheduler =None
        tensorboard_callback = None
        self.callbacks = []
        if(self.lr_decay_policy == 'poly'):
            scheduler = schedule(self.lr,self.lr_decay_power,self.epochs)#PolynomialDecay(maxEpochs=100, initAlpha=0.01, power=self.lr_decay_power)
        # initializing tensorboard
#        shutil.rmtree(self.logdir+"/train/")
        os.makedirs(self.logdir, exist_ok=True)
        tensorboard_callback = TensorBoard(log_dir=self.logdir)
        
        if(scheduler):
            print("learning rate scheduler added\n\n")
            self.callbacks.append(LearningRateScheduler(scheduler))
        if(tensorboard_callback):
            print("tensor boarda callback added  with folder " + self.logdir +"\n\n")
            self.callbacks.append(tensorboard_callback)
        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="/home/essys/projects/segmentation/checkpoints/" + self.train_id + "/cp-{epoch:04d}.ckpt", 
            verbose=0, 
            save_weights_only=True,
            period=50)
        self.callbacks.append(cp_callback)
        diagnoser = ModelDiagonoser(self.training_generator, self.batch_size, self.n_samples, self.logdir,self.input_shape,self.n_classes)
        self.callbacks.append(diagnoser)
#        image_callback = TensorBoardImage('tag', self.logdir, self.model.outp)
    def init_metrics(self):
        assert not len(self.metrics_names) == 0
        self.metrics = []
        for i in range(len(self.metrics_names)):
            if(self.metrics_names[i] == 'accuracy'):
                self.metrics.append('accuracy')
                # m = tf.keras.metrics.MeanIoU(num_classes=5)
                # self.metrics.append(m)
    def init_losses(self, use_output_layer_names=False):
        # if(self.model_out_layer_names is None):
        #     assert not len(self.loss_names) == 0, " number of loss functions cannot be zero"
        # else:
        #     # print(self.model_out_layer_names, self.loss_names)
        #     assert len(self.loss_names) == len(self.model_out_layer_names), "in " + self.model_name + \
        #         " number of loss fs should be equal to number of output layers of the model"
#        if(len(self.loss_names) > 1):
        if(use_output_layer_names):
            self.losses = {}
            for i in range(len(self.model_out_layer_names)):
                self.losses[self.model_out_layer_names[i]] = self.get_loss(self.loss_names[i])
        else:
            self.losses= self.get_loss(self.loss_names[0])

    def get_loss(self, name):
        if(name == 'categorical_crossentropy'):
            return 'categorical_crossentropy'
        elif(name == 'iou_categorical_crossentropy'):
            return IOULOSS(self.n_classes, self.batch_size, self.input_shape, self.class_weights)
        elif(name == 'categorical_crossentropy_weighted'):
            return categorical_crossentropy_weighted(self.class_weights)
    def init_model(self,):
        if(self.model_name == "PSPNet"):
            self.model, self.model_out_layer_names = PSPNet(backbone_name=self.backbone,input_shape=self.input_shape, classes=self.n_classes,encoder_weights=self.pretrained_encoder_weights,
             encoder_freeze=self.transfer_learning,training=self.training)
            try:
                # The model weights (that are considered the best) are loaded into the model.
                self.model.load_weights('/home/essys/projects/segmentation/checkpoints/' + self.train_id + "/")
            except:
                print('could not find saved model')
        if(self.model_name == "Bisenet_V2"):
            self.model = BisenetV2Model(train_op= optimizers.Adam(self.lr), input_shape=self.input_shape,classes=self.n_classes, batch_size=self.batch_size, class_weights=self.class_weights) #BisenetV2(input_shape=self.input_shape,classes=self.n_classes)    
            try:
                checkpoint_dir = '/home/essys/projects/segmentation/checkpoints/' + self.train_id + "/"
                latest = tf.train.latest_checkpoint(checkpoint_dir)
                print(latest + " is found model\n\n")
                # The model weights (that are considered the best) are loaded into the model.
                self.model.model.load_weights(latest)                
            except:
                print('could not find saved model')
        
    def add_weight_decay(self,):
        # https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.DepthwiseConv2D):
                layer.add_loss(keras.regularizers.l2(self.weight_decay)(layer.depthwise_kernel))
            elif isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                layer.add_loss(keras.regularizers.l2(self.weight_decay)(layer.kernel))
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                layer.add_loss(keras.regularizers.l2(self.weight_decay)(layer.bias))

    def init_optimizer(self,):
        if(self.optimizer_name == "Adam"):
            self.optimizer = optimizers.Adam(lr=self.lr)
        if(self.optimizer_name == "SGD"):
            self.optimizer = optimizers.SGD(self.lr, momentum=self.momentum)




if __name__ == "__main__":
    builder = Builder()
    builder.demo_train()