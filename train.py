# -*- coding: utf-8 -*-

#!pip install asyncstdlib
!pip install gdown

import warnings
warnings.filterwarnings('ignore')

import os
import gc
import cv2
import gdown
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# from asyncstdlib.builtins import map
from skimage.io import imread
from sklearn.model_selection import train_test_split


gc.enable()

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
strategy = tf.distribute.MirroredStrategy()

# Train and test directories
train_image_dir = '/airbus-ship-detection/train_v2/'
test_image_dir = '/airbus-ship-detection/test_v2/'
LABELS = '/airbus-ship-detection/train_ship_segmentations_v2.csv'


# Parameters / CONST
ORIG_IMG_SIZE = 768
BATCH_SIZE = 4                 # Train batch size (4)
EDGE_CROP = 16                 # While building the model (16)
NB_EPOCHS = 1                  # Training epochs. (5)
GAUSSIAN_NOISE = 0.1           # To be used in a layer in the model
UPSAMPLE_MODE = 'DECONV'       # SIMPLE ==> UpSampling2D, else DECONV ==> Conv2DTranspose
NET_SCALING = None             # Downsampling inside the network  (None)
IMG_SCALING = (1, 1)           # Downsampling in preprocessing (1, 1)
VALID_IMG_COUNT = 256          # Valid batch size (256)
MAX_TRAIN_STEPS = 128          # Maximum number of steps_per_epoch in training (200)

train_images = os.listdir(train_image_dir)
train_images.sort()
print(f"Total of {len(train_images)} images in train_dir.\nFirst 5 train_images list: {train_images[:5]}")

# Train ships segmented masks
masks = pd.read_csv(LABELS)

# Define functions to do these tasks for all the training images

def rle_decode(mask_rle, shape=(ORIG_IMG_SIZE,ORIG_IMG_SIZE)):
    '''
    Input arguments -
    mask_rle: Mask of one ship in the train image
    shape: Output shape of the image array
    '''
    s = mask_rle.split()                                                               # Split the mask of each ship that is in RLE format
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]     # Get the start pixels and lengths for which image has ship
    ends = starts + lengths - 1                                                        # Get the end pixels where we need to stop
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)                                  # A 1D vec full of zeros of size = 768*768
    for lo, hi in zip(starts, ends):                                                   # For each start to end pixels where ship exists
        img[lo:hi+1] = 1                                                               # Fill those values with 1 in the main 1D vector
    '''
    Returns -
    Transposed array of the mask: Contains 1s and 0s. 1 for ship and 0 for background
    '''
    return img.reshape(shape).T

def masks_as_image(in_mask_list):
    '''
    Input -
    in_mask_list: List of the masks of each ship in one whole training image
    '''
    all_masks = np.zeros((768, 768), dtype = np.int16)                                 # Creating 0s for the background
    for mask in in_mask_list:                                                          # For each ship rle data in the list of mask rle
        if isinstance(mask, str):                                                      # If the datatype is string
            all_masks += rle_decode(mask)                                              # Use rle_decode to create one mask for whole image
    '''
    Returns -
    Full mask of the training image whose RLE data has been passed as an input
    '''
    return np.expand_dims(all_masks, -1)

# Add a new feature to the masks data frame named as ship. If Encoded pixel in any row is a string, there is a ship else there isn't.

masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

# Making a new df with unique image IDs where we are summing up the ship counts

unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids.index+=1 # Incrimenting all the index by 1

# Adding two new features to unique_img_ids df. If ship exists in image, val is 1 else 0 (vec form)

unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)

# Check the size of the files. Will take a lot of time to run as there are loads of files

#Uncomment on PROD
# unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: os.stat(os.path.join(train_image_dir, c_img_id)).st_size/1024)
# '''os.stat is used to get status of the specified path. Here, st_size represents size of the file in bytes. Converting it into kB!'''

# import pickle
# with open('unique_img_ids.pkl', 'wb') as fp:
#     pickle.dump(unique_img_ids, fp)
#     print('unique_img_ids dictionary saved successfully dumped')

#COMMENT CODE ON PROD

URL_PKL = "https://drive.google.com/uc?id=1t5nKU3jF3GHn9PgXfG0MLLD0QC2xqSCd"
gdown.download(url=URL_PKL, output="unique_img_ids.pkl", quiet=False)

with open('unique_img_ids.pkl', 'rb') as fp:
    unique_img_ids = pickle.load(fp)

# Keep the files whose size > 35 kB

unique_img_ids = unique_img_ids[unique_img_ids.file_size_kb > 35]

# Retrive the old masks data frame

masks.drop(['ships'], axis=1, inplace=True)
masks.index+=1

# Train - Test split

train_ids, valid_ids = train_test_split(unique_img_ids, test_size = 0.3, stratify = unique_img_ids['ships'])

# Create train df

train_df = pd.merge(masks, train_ids)

# Create test df

valid_df = pd.merge(masks, valid_ids)

# Clipping the MAX value of grouped_ship_count to be 7, minimum to be 0

train_df['grouped_ship_count'] = train_df.ships.map(lambda x: (x+1)//2).clip(0,7)

# Random Under-Sampling ships

def sample_ships(in_df, base_rep_val=2000):
    '''
    Input Args:
    in_df - dataframe we want to apply this function
    base_val - random sample of this value to be taken from the data frame
    '''
    if in_df['ships'].values[0]==0:
        return in_df.sample(base_rep_val//2)  # Random 1000 samples taken whose ship count is 0 in an image
    else:
        return in_df.sample(base_rep_val)     # Random 2000 samples taken whose ship count is not 0 in an image

# Creating groups of ship counts and applying the sample_ships functions to randomly undersample ships

balanced_train_df = train_df.groupby('grouped_ship_count').apply(sample_ships)

# Image and Mask Generator

def make_image_gen(in_df, batch_size = BATCH_SIZE):
    '''
    Inputs -
    in_df - data frame on which the function will be applied
    batch_size - number of training examples in one iteration
    '''
    all_batches = list(in_df.groupby('ImageId'))                             # Group ImageIds and create list of that dataframe
    out_rgb = []                                                             # Image list
    out_mask = []                                                            # Mask list
    while True:                                                              # Loop for every data
        np.random.shuffle(all_batches)                                       # Shuffling the data
        for c_img_id, c_masks in all_batches:                                # For img_id and msk_rle in all_batches
            rgb_path = os.path.join(train_image_dir, c_img_id)               # Get the img path
            c_img = imread(rgb_path)                                         # img array
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)         # Create mask of rle data for each ship in an img
            out_rgb += [c_img]                                               # Append the current img in the out_rgb / img list
            out_mask += [c_mask]                                             # Append the current mask in the out_mask / mask list
            if len(out_rgb)>=batch_size:                                     # If length of list is more or equal to batch size then
                yield np.stack(out_rgb)/255.0, np.stack(out_mask)            # Yeild the scaled img array (b/w 0 and 1) and mask array (0 for bg and 1 for ship)
                out_rgb, out_mask=[], []                                     # Empty the lists to create another batch

# Generate train data

train_gen = make_image_gen(balanced_train_df)

# Image and Mask

train_x, train_y = next(train_gen)

# Prepare validation data

valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))

# Augmenting Data

from keras.preprocessing.image import ImageDataGenerator

# Preparing image data generator arguments

dg_args = dict(rotation_range = 15,            # Degree range for random rotations
               horizontal_flip = True,         # Randomly flips the inputs horizontally
               vertical_flip = True,           # Randomly flips the inputs vertically
               data_format = 'channels_last')  # channels_last refer to (batch, height, width, channels)

image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)

def create_aug_gen(in_gen, seed = None):
    '''
    Takes in -
    in_gen - train data generator, seed value
    '''
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))  # Randomly assign seed value if not provided
    for in_x, in_y in in_gen:                                                    # For imgs and msks in train data generator
        seed = 12                                                                # Seed value for imgs and msks must be same else augmentation won't be same

        # Create augmented imgs
        g_x = image_gen.flow(255*in_x,                                           # Inverse scaling on imgs for augmentation
                             batch_size = in_x.shape[0],                         # batch_size = 3
                             seed = seed,                                        # Seed
                             shuffle=True)                                       # Shuffle the data

        # Create augmented masks
        g_y = label_gen.flow(in_y,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)

        '''Yeilds - augmented scaled imgs and msks array'''
        yield next(g_x)/255.0, next(g_y)

# Augment train data

cur_gen = create_aug_gen(train_gen, seed = 42)
t_x, t_y = next(cur_gen)

# Block all the garbage that has been generated
gc.collect()

# Build U-Net model

from keras import models, layers

# Conv2DTranspose upsampling
def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
# Upsampling without Conv2DTranspose
def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)

# Upsampling method choice
if UPSAMPLE_MODE=='DECONV':
    upsample=upsample_conv
else:
    upsample=upsample_simple

# Building the layers of UNET
input_img = layers.Input(t_x.shape[1:], name = 'RGB_Input')
pp_in_layer = input_img

# If NET_SCALING is defined then do the next step else continue ahead
if NET_SCALING is not None:
    pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

# To avoid overfitting and fastening the process of training
pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer) # Useful to mitigate overfitting
pp_in_layer = layers.BatchNormalization()(pp_in_layer)          # Allows using higher learning rate without causing problems with gradients


## Downsample (C-->C-->MP)

c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (pp_in_layer)
c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = layers.MaxPooling2D((2, 2)) (c1)

c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = layers.MaxPooling2D((2, 2)) (c2)

c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = layers.MaxPooling2D((2, 2)) (c3)

c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)


c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

## Upsample (U --> Concat --> C --> C)

u6 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = upsample(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)

if NET_SCALING is not None:
    d = layers.UpSampling2D(NET_SCALING)(d)

seg_model = models.Model(inputs=[input_img], outputs=[d])

# Compute DICE coefficient, loss with BCE and compile the model

import keras.backend as K
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy

# Dice coeff
def dice_coef(y_true, y_pred, smooth=1):
    y_pred = tf.dtypes.cast(y_pred, tf.int32)
    y_true = tf.dtypes.cast(y_true, tf.int32)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])                         # int = y_true ∩ y_pred
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])           # un = y_true_flatten ed ∪ y_pred_flattened
    return K.mean((2 * intersection + smooth) / (union + smooth), axis=0)       # dice = 2 * int + 1 / un + 1

# Dice with BCE
def dice_p_bce(y_true, y_pred, alpha=1e-2):
    """combine DICE and BCE"""
    combo_loss = alpha*binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
    return combo_loss

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

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

def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    focal_loss_fixed = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

# Compile the model (learning_rate=1e-4)

opt = Adam(learning_rate=1e-4, weight_decay=1e-6)
seg_model.compile(optimizer=opt, loss=Combo_loss, metrics=[dice_coef, 'binary_accuracy'])

# Preparing Callbacks

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

# Best model weights
weight_path="{}_weights.best.hdf5".format('seg_model')

# Monitor validation dice coeff and save the best model weights
checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                             save_best_only=True, mode='max', save_weights_only = True)

# Reduce Learning Rate on Plateau
reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=3,
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)

# Stop training once there is no improvement seen in the model
early = EarlyStopping(monitor="val_dice_coef", mode="max",
                      patience=15) # probably needs to be more patient, but VPS time is limited

# Callbacks ready
callbacks_list = [checkpoint, early, reduceLROnPlat]

from keras import backend as K
gc.collect()
K.clear_session()

# Finalizing steps per epoch
step_count = min(MAX_TRAIN_STEPS, balanced_train_df.shape[0]//BATCH_SIZE)

# Final augmented data being used in training
aug_gen = create_aug_gen(make_image_gen(balanced_train_df))

# Save loss history while training
loss_history = [seg_model.fit_generator(aug_gen,
                             steps_per_epoch=step_count,
                             epochs=NB_EPOCHS,
                             validation_data=(valid_x, valid_y),
                             callbacks=callbacks_list, workers=1)]

# Save the weights to load it later for test data
seg_model.load_weights(weight_path)
seg_model.save('seg_unet_model.h5')