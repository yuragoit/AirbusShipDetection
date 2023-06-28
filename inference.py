# -*- coding: utf-8 -*-

#!pip install asyncstdlib --quiet
!pip install gdown --quiet
import warnings
warnings.filterwarnings('ignore')


# Commented out IPython magic to ensure Python compatibility.
import os
import cv2
import gdown
import tensorflow as tf
import numpy as np
import keras.backend as K


from tensorflow import keras
from matplotlib import pyplot as plt
%matplotlib inline

# CONST
test_image_dir = '/airbus-ship-detection/test_v2/'
URL_H5 = "https://drive.google.com/uc?id=1FhICkeGn6GcNXWTDn1s83ctC-6Mo1UXk"
IMG_SCALING = (1, 1)
test_imgs = ['00dc34840.jpg', '00c3db267.jpg', '00aa79c47.jpg', '011ee8cd9.jpg']

gdown.download(url=URL_H5, output="seg_unet_model.h5", quiet=False)

def gen_pred(test_dir, img, model):
    rgb_path = os.path.join(test_image_dir,img)
    img = cv2.imread(rgb_path)
    img = img[::IMG_SCALING[0], ::IMG_SCALING[1]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv2.imread(rgb_path), pred

#Custom objects for model

def Combo_loss(y_true, y_pred, eps=1e-9, smooth=1):
    targets = tf.dtypes.cast(K.flatten(y_true), tf.float32)
    inputs = tf.dtypes.cast(K.flatten(y_pred), tf.float32)

    intersection = K.sum(targets * inputs)
    dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    inputs = K.clip(inputs, eps, 1.0 - eps)
    out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
    return combo

def dice_coef(y_true, y_pred, smooth=1):
    y_pred = tf.dtypes.cast(y_pred, tf.int32)
    y_true = tf.dtypes.cast(y_true, tf.int32)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])                         # int = y_true ∩ y_pred
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])           # un = y_true_flatten ed ∪ y_pred_flattened
    return K.mean((2 * intersection + smooth) / (union + smooth), axis=0)     # dice = 2 * int + 1 / un + 1

def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    focal_loss_fixed = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

# Load pretrained model

#for kaggle users
#seg_model = keras.models.load_model('seg_unet_model.h5', custom_objects={'Combo_loss': Combo_loss, 'dice_coef': dice_coef})
#for local users
seg_model = keras.models.load_model('models/seg_unet_model.h5', custom_objects={'Combo_loss': Combo_loss, 'dice_coef': dice_coef, 'focal_loss_fixed': focal_loss_fixed})

# Visualize some predictions

rows = 1
columns = 2
for i in range(len(test_imgs)):
    img, pred = gen_pred(test_image_dir, test_imgs[i], seg_model)
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(pred, interpolation='catrom')
    plt.axis('off')
    plt.title("Prediction")