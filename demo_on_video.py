# -*- coding: utf-8 -*-

import os
from tensorflow.keras import models 

from tensorflow.keras import backend as K
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = '1'
import numpy as np
import time
from models.BisenetV3 import  BisenetV2Model
import cv2

c = 3
h = 288
w = 800

model_path ="/home/essys/projects/segmentation/checkpoints/bisnet_v2_test/cp-epoch-131-step-273499.ckpt"
model_cl  = BisenetV2Model(None, (h,w,1),classes=2,training=True,batch_size=1) #BisenetV2((h,w,3),classes=2,training=True)
model_cl = model_cl.model
model_cl.load_weights(model_path)
logits = model_cl.output[0]
logits = tf.argmax(logits, -1)
model = models.Model(model_cl.input, logits)
cam = cv2.VideoCapture("test.mp4")

start1= time.time()
counter = 0
while(True):
    counter =counter + 1
    start = time.time()
    ret = cam.grab()
    if(counter % 4==0):
        ret, frame = cam.retrieve()
    else:
        continue
    # frame = frame[150:,...]
    img = cv2.resize(frame, (800,288),interpolation =cv2.INTER_NEAREST)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # pred = model(np.expand_dims(gray, 0), training=False)
    pred = model.predict(np.expand_dims(gray/255, 0))
    # print(np.shape(pred))
    
    
    cv2.imshow("pred", np.uint8(pred[0]) * 255)
    cv2.imshow("image", gray)
    key = cv2.waitKey(3)
    if(key == 27):
        cv2.destroyAllWindows()
        break
    print((time.time()-start))