import os
from tensorflow.keras import models 
from tensorflow.keras import layers

from tensorflow.keras import backend as K
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = '1'
import numpy as np
import time
from models.backbones import ResNet50, ResNet152V2, DRN50, DRN101, DRN152
from models import PSPNet, BisenetV2Model

#model ,_=  PSPNet(backbone_name='resnet50',input_shape=(241,441,3),classes=5,psp_dropout=0.1)
## model =  DRN50(aux=False, include_top=False,input_shape=(713,713,3))
#l1 = model.layers[-1]
#l2 = model.layers[-2]
#
#print(model.summary())
#print(l1)
#print(l2)
## print(model.variables)
## for i in range(len(model.layers)):
## print(model.get_layer('conv4_block23_out').name)
## print(model.get_layer('conv4_block23_out').variables)
#

def get_flops(model_h5_path="model.h5"):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
        
            return flops.total_float_ops


h =488
w= 800
c = 3
# input1 = layers.Input(shape=[h,w,c])
# input2 = layers.Input(shape=[int(h/4),int(w/4),c])

# x = AggregationLayer(out_channels=c)(input1, input2)
# x = SemanticBranch()(input1)
# model = models.Model(input1, x)

# policy =tf.keras.mixed_precision.experimental.Policy('float16')
# tf.keras.mixed_precision.experimental.set_policy(policy)
# backend.set_floatx('float16')
# backend.set_epsilon(1e-4) #default is 1e-7
# tf.keras.backend.set_floatx('float16')
if(1):
    model_cl  = BisenetV2Model(None, (h,w,1),classes=2,training=True,batch_size=1) #BisenetV2((h,w,3),classes=2,training=True)
    model_cl = model_cl.model
    model_cl.save_weights("model.h5")
    start1= time.time()
    for i in range(100):
        start = time.time()
        img = np.zeros([1, h,w,1])
        
        pred = model_cl(img, training=False)
        print((time.time()-start))
print('\n\n\nstarting without auxillarry parts\n\n\n')
# model_cl  = BisenetV2((h,w,3),classes=2,training=False)
#model_cl = model_cl.init_model()
model = models.Model(model_cl.input, model_cl.output[0])
# model.load_weights("model.h5")

for i in range(100):
    start = time.time()
    img = np.zeros([1, h,w,1])
    
    pred = model(img, training=False)
    print((time.time()-start))

# 
# print('\n\n')
# print(get_flops())
# print('\n\n')
# print(model.summary())
